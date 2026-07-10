//  Interest Rate Curve Construction
//
//  Builds discount curves from market quotes (cash deposits and par swaps),
//  prices a new swap against them, and computes analytic risk back to the
//  quotes. Covers Q1 (curve construction + interpolation), Q2.1 (pricing),
//  Q2.2 (analytic risk) and Q3 (generic, extensible design).
//
//  The design keeps interpolation, instruments, calibration and risk as
//  separate concerns: Interpolator (Strategy), the PricingInstrument /
//  CalibrationInstrument interfaces, and the generic CurveCalibrator and
//  RiskEngine. A new instrument or interpolation scheme slots in by
//  implementing one interface, without touching the rest. Each type is
//  documented at its definition below; the accompanying write-up has the
//  full architecture discussion.
//
//  Conventions (per problem statement): 1W = 7d, 1M = 30d, 1Y = 360d,
//    DCF(t2,t1) = (t2 - t1) / 360
//    DF(t)      = 1 / (1 + r(t) * DCF(t,0))
//    F(t1,t2)   = ( DF(t1)/DF(t2) - 1 ) / DCF(t2,t1)
//  Interpolation is done on log(DF) for numerical stability and positivity.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace curve {

// Day-count denominator (ACT/360 with the simplified calendar).
constexpr double kDayCountBasis = 360.0;

// Day-count fraction between two times expressed in days.
inline double dayCountFraction(double daysEnd, double daysStart) {
    return (daysEnd - daysStart) / kDayCountBasis;
}

// Tenor parsing: "1D","2W","3M","5Y" -> number of days (simplified calendar).
inline double tenorToDays(const std::string& tenor) {
    if (tenor.size() < 2)
        throw std::invalid_argument("Invalid tenor: '" + tenor + "'");
    const char unit = static_cast<char>(std::toupper(tenor.back()));
    const std::string numberPart = tenor.substr(0, tenor.size() - 1);
    double n = 0.0;
    try {
        n = std::stod(numberPart);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid tenor magnitude: '" + tenor + "'");
    }
    switch (unit) {
        case 'D': return n;
        case 'W': return n * 7.0;
        case 'M': return n * 30.0;
        case 'Y': return n * 360.0;
        default:
            throw std::invalid_argument("Unknown tenor unit in '" + tenor + "'");
    }
}

// Frequency parsing for swap legs: "1m","3m","6m","12m" -> period in days.
inline double frequencyToDays(const std::string& freq) {
    if (freq.size() < 2)
        throw std::invalid_argument("Invalid frequency: '" + freq + "'");
    const char unit = static_cast<char>(std::tolower(freq.back()));
    if (unit != 'm')
        throw std::invalid_argument("Unsupported frequency unit in '" + freq + "'");
    double n = 0.0;
    try {
        n = std::stod(freq.substr(0, freq.size() - 1));
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid frequency magnitude: '" + freq + "'");
    }
    if (n <= 0.0)
        throw std::invalid_argument("Frequency must be positive: '" + freq + "'");
    return n * 30.0;
}

// Interpolation layer

// A single calibrated point on a discount curve: time (days) and log(DF).
struct CurveNode {
    double t;
    double logDf;
};

// Strategy interface for interpolating log(DF).
//
// Every scheme also reports the exact linear weights w_k such that
//      interpolateLogDf(t) == sum_k w_k * y_k         (y_k = node log-DF)
// Since all the schemes here are linear in the node log-DFs, these weights
// are exactly the partial derivatives d ln DF(t)/d y_k, which is what lets the
// risk engine stay fully analytic (no bump-and-revalue).
class Interpolator {
public:
    virtual ~Interpolator() = default;
    virtual double interpolateLogDf(const std::vector<CurveNode>& nodes,
                                    double t) const = 0;
    virtual std::vector<double> logDfWeights(const std::vector<CurveNode>& nodes,
                                             double t) const = 0;
    virtual std::unique_ptr<Interpolator> clone() const = 0;
    virtual std::string name() const = 0;

protected:
    // Index i with nodes[i].t <= t <= nodes[i+1].t. Out-of-range t falls back
    // to the nearest boundary interval (boundary extrapolation on log-DF).
    static std::size_t locateInterval(const std::vector<CurveNode>& nodes,
                                       double t) {
        const std::size_t n = nodes.size();
        if (t <= nodes.front().t) return 0;
        if (t >= nodes.back().t) return n - 2;
        std::size_t lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const std::size_t mid = (lo + hi) / 2;
            if (nodes[mid].t <= t) lo = mid; else hi = mid;
        }
        return lo;
    }
};

// (A) Linear interpolation on log(DF):
//   Ln(DF) = Ln(DF_{i-1}) + (t-t_{i-1})/(t_i-t_{i-1})*(Ln(DF_i)-Ln(DF_{i-1}))
class LinearInterpolator final : public Interpolator {
public:
    double interpolateLogDf(const std::vector<CurveNode>& nodes,
                            double t) const override {
        if (nodes.empty())
            throw std::runtime_error("Cannot interpolate on an empty curve");
        if (nodes.size() == 1) return nodes.front().logDf;
        const std::size_t i = locateInterval(nodes, t);
        const double dt = nodes[i + 1].t - nodes[i].t;
        if (std::abs(dt) < 1e-12) return nodes[i].logDf;
        const double w = (t - nodes[i].t) / dt;
        return nodes[i].logDf + w * (nodes[i + 1].logDf - nodes[i].logDf);
    }

    std::vector<double> logDfWeights(const std::vector<CurveNode>& nodes,
                                     double t) const override {
        std::vector<double> w(nodes.size(), 0.0);
        if (nodes.empty()) return w;
        if (nodes.size() == 1) { w[0] = 1.0; return w; }
        const std::size_t i = locateInterval(nodes, t);
        const double dt = nodes[i + 1].t - nodes[i].t;
        if (std::abs(dt) < 1e-12) { w[i] = 1.0; return w; }
        const double frac = (t - nodes[i].t) / dt;
        w[i] = 1.0 - frac;
        w[i + 1] = frac;
        return w;
    }

    std::unique_ptr<Interpolator> clone() const override {
        return std::make_unique<LinearInterpolator>(*this);
    }
    std::string name() const override { return "Linear"; }
};

// (B) Averaged-Quadratic interpolation on log(DF).
//   For t in [t_i, t_{i+1}] (not the first interval) blend the quadratic
//   through (t_{i-1},t_i,t_{i+1}) and through (t_i,t_{i+1},t_{i+2}):
//     Ln(DF) = (t_{i+1}-t)/(t_{i+1}-t_i)*Q_i + (t-t_i)/(t_{i+1}-t_i)*Q_{i+1}
//   Documented assumptions:
//     * First interval [t_0,t_1] : linear (per the spec note).
//     * Last interval (no t_{i+2}) : fall back to the single available backward
//       quadratic Q_i, since the forward quadratic isn't defined there.
class AveragedQuadraticInterpolator final : public Interpolator {
public:
    double interpolateLogDf(const std::vector<CurveNode>& nodes,
                            double t) const override {
        const std::size_t n = nodes.size();
        if (n == 0) throw std::runtime_error("Cannot interpolate on empty curve");
        if (n == 1) return nodes.front().logDf;
        if (n == 2) return linear(nodes[0], nodes[1], t);
        const std::size_t i = locateInterval(nodes, t);
        if (i == 0) return linear(nodes[0], nodes[1], t);     // first: linear
        const bool hasForward = (i + 2 <= n - 1);
        const double qi = quadratic(nodes[i - 1], nodes[i], nodes[i + 1], t);
        if (!hasForward) return qi;                           // last: backward only
        const double qi1 = quadratic(nodes[i], nodes[i + 1], nodes[i + 2], t);
        const double span = nodes[i + 1].t - nodes[i].t;
        if (std::abs(span) < 1e-12) return nodes[i].logDf;
        const double wLeft = (nodes[i + 1].t - t) / span;
        const double wRight = (t - nodes[i].t) / span;
        return wLeft * qi + wRight * qi1;
    }

    std::vector<double> logDfWeights(const std::vector<CurveNode>& nodes,
                                     double t) const override {
        const std::size_t n = nodes.size();
        std::vector<double> w(n, 0.0);
        if (n == 0) return w;
        if (n == 1) { w[0] = 1.0; return w; }
        if (n == 2) { linearW(nodes[0], nodes[1], t, 0, 1, w); return w; }
        const std::size_t i = locateInterval(nodes, t);
        if (i == 0) { linearW(nodes[0], nodes[1], t, 0, 1, w); return w; }
        const bool hasForward = (i + 2 <= n - 1);
        if (!hasForward) {
            quadW(nodes[i - 1], nodes[i], nodes[i + 1], t, i - 1, i, i + 1, 1.0, w);
            return w;
        }
        const double span = nodes[i + 1].t - nodes[i].t;
        if (std::abs(span) < 1e-12) { w[i] = 1.0; return w; }
        const double wLeft = (nodes[i + 1].t - t) / span;
        const double wRight = (t - nodes[i].t) / span;
        quadW(nodes[i - 1], nodes[i], nodes[i + 1], t, i - 1, i, i + 1, wLeft, w);
        quadW(nodes[i], nodes[i + 1], nodes[i + 2], t, i, i + 1, i + 2, wRight, w);
        return w;
    }

    std::unique_ptr<Interpolator> clone() const override {
        return std::make_unique<AveragedQuadraticInterpolator>(*this);
    }
    std::string name() const override { return "AveragedQuadratic"; }

private:
    static double linear(const CurveNode& a, const CurveNode& b, double t) {
        const double dt = b.t - a.t;
        if (std::abs(dt) < 1e-12) return a.logDf;
        return a.logDf + (t - a.t) / dt * (b.logDf - a.logDf);
    }
    static void linearW(const CurveNode& a, const CurveNode& b, double t,
                        std::size_t ia, std::size_t ib, std::vector<double>& w) {
        const double dt = b.t - a.t;
        if (std::abs(dt) < 1e-12) { w[ia] += 1.0; return; }
        const double frac = (t - a.t) / dt;
        w[ia] += 1.0 - frac;
        w[ib] += frac;
    }
    static double quadratic(const CurveNode& p0, const CurveNode& p1,
                            const CurveNode& p2, double t) {
        const double x0 = p0.t, x1 = p1.t, x2 = p2.t;
        const double l0 = ((t - x1) * (t - x2)) / ((x0 - x1) * (x0 - x2));
        const double l1 = ((t - x0) * (t - x2)) / ((x1 - x0) * (x1 - x2));
        const double l2 = ((t - x0) * (t - x1)) / ((x2 - x0) * (x2 - x1));
        return p0.logDf * l0 + p1.logDf * l1 + p2.logDf * l2;
    }
    static void quadW(const CurveNode& p0, const CurveNode& p1,
                      const CurveNode& p2, double t, std::size_t i0,
                      std::size_t i1, std::size_t i2, double scale,
                      std::vector<double>& w) {
        const double x0 = p0.t, x1 = p1.t, x2 = p2.t;
        w[i0] += scale * ((t - x1) * (t - x2)) / ((x0 - x1) * (x0 - x2));
        w[i1] += scale * ((t - x0) * (t - x2)) / ((x1 - x0) * (x1 - x2));
        w[i2] += scale * ((t - x0) * (t - x1)) / ((x2 - x0) * (x2 - x1));
    }
};

// (C) Natural cubic spline on log(DF), added for Q3.
//
// I included this mainly to show the system is extensible: it dropped in
// without touching the curve, calibrator, pricer, or risk engine. A natural
// cubic spline (zero second derivative at the ends) is fit to (t, logDf).
// Since the spline value is linear in the node log-DFs, the exact interpolation
// weights (and so the analytic risk) come for free by solving the same spline
// system against the unit basis vectors.
class CubicSplineInterpolator final : public Interpolator {
public:
    double interpolateLogDf(const std::vector<CurveNode>& nodes,
                            double t) const override {
        const std::size_t n = nodes.size();
        if (n == 0) throw std::runtime_error("Cannot interpolate on empty curve");
        if (n == 1) return nodes.front().logDf;
        if (n == 2) {
            const double dt = nodes[1].t - nodes[0].t;
            return nodes[0].logDf + (t - nodes[0].t) / dt *
                                        (nodes[1].logDf - nodes[0].logDf);
        }
        std::vector<double> y(n);
        for (std::size_t k = 0; k < n; ++k) y[k] = nodes[k].logDf;
        return evaluate(nodes, y, t);
    }

    std::vector<double> logDfWeights(const std::vector<CurveNode>& nodes,
                                     double t) const override {
        const std::size_t n = nodes.size();
        std::vector<double> w(n, 0.0);
        if (n == 0) return w;
        if (n == 1) { w[0] = 1.0; return w; }
        // Linearity: weight_k = spline value at t built from the unit basis e_k.
        for (std::size_t k = 0; k < n; ++k) {
            std::vector<double> e(n, 0.0);
            e[k] = 1.0;
            w[k] = (n == 2)
                       ? linearBasis(nodes, k, t)
                       : evaluate(nodes, e, t);
        }
        return w;
    }

    std::unique_ptr<Interpolator> clone() const override {
        return std::make_unique<CubicSplineInterpolator>(*this);
    }
    std::string name() const override { return "CubicSpline"; }

private:
    static double linearBasis(const std::vector<CurveNode>& nodes,
                              std::size_t k, double t) {
        const double dt = nodes[1].t - nodes[0].t;
        const double frac = (t - nodes[0].t) / dt;
        if (k == 0) return 1.0 - frac;
        return frac;
    }

    // Evaluate a natural cubic spline with knots nodes[].t and values y at t.
    static double evaluate(const std::vector<CurveNode>& nodes,
                           const std::vector<double>& y, double t) {
        const std::size_t n = nodes.size();
        std::vector<double> h(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) h[i] = nodes[i + 1].t - nodes[i].t;

        // Solve tridiagonal system for second derivatives m[] (natural: ends 0).
        std::vector<double> m(n, 0.0), rhs(n, 0.0), lower(n, 0.0),
            diag(n, 1.0), upper(n, 0.0);
        for (std::size_t i = 1; i + 1 < n; ++i) {
            lower[i] = h[i - 1];
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }
        // Thomas algorithm.
        for (std::size_t i = 1; i < n; ++i) {
            const double factor = lower[i] / diag[i - 1];
            diag[i] -= factor * upper[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }
        m[n - 1] = (diag[n - 1] != 0.0) ? rhs[n - 1] / diag[n - 1] : 0.0;
        for (std::size_t i = n - 1; i-- > 0;)
            m[i] = (diag[i] != 0.0) ? (rhs[i] - upper[i] * m[i + 1]) / diag[i] : 0.0;

        // Clamp to range (flat extrapolation) then evaluate the cubic piece.
        double x = t;
        if (x <= nodes.front().t) x = nodes.front().t;
        if (x >= nodes.back().t) x = nodes.back().t;
        std::size_t i = 0;
        while (i + 1 < n && x > nodes[i + 1].t) ++i;
        const double dx = nodes[i + 1].t - x;
        const double dxp = x - nodes[i].t;
        const double hi = h[i];
        return m[i] * dx * dx * dx / (6.0 * hi)
             + m[i + 1] * dxp * dxp * dxp / (6.0 * hi)
             + (y[i] / hi - m[i] * hi / 6.0) * dx
             + (y[i + 1] / hi - m[i + 1] * hi / 6.0) * dxp;
    }
};

// Factory for interpolators (Factory pattern). New scheme => one line here.
enum class InterpolationMethod { Linear, AveragedQuadratic, CubicSpline };

class InterpolatorFactory {
public:
    static std::unique_ptr<Interpolator> create(InterpolationMethod m) {
        switch (m) {
            case InterpolationMethod::Linear:
                return std::make_unique<LinearInterpolator>();
            case InterpolationMethod::AveragedQuadratic:
                return std::make_unique<AveragedQuadraticInterpolator>();
            case InterpolationMethod::CubicSpline:
                return std::make_unique<CubicSplineInterpolator>();
        }
        throw std::invalid_argument("Unknown interpolation method");
    }
    static std::unique_ptr<Interpolator> create(const std::string& name) {
        if (name == "Linear") return create(InterpolationMethod::Linear);
        if (name == "AveragedQuadratic")
            return create(InterpolationMethod::AveragedQuadratic);
        if (name == "CubicSpline") return create(InterpolationMethod::CubicSpline);
        throw std::invalid_argument("Unknown interpolation method: " + name);
    }
};

// Discount curve
class DiscountCurve {
public:
    explicit DiscountCurve(std::unique_ptr<Interpolator> interp)
        : interp_(std::move(interp)) {
        if (!interp_) throw std::invalid_argument("Interpolator must not be null");
    }
    DiscountCurve(const DiscountCurve& o)
        : nodes_(o.nodes_), interp_(o.interp_->clone()) {}
    DiscountCurve& operator=(const DiscountCurve& o) {
        if (this != &o) { nodes_ = o.nodes_; interp_ = o.interp_->clone(); }
        return *this;
    }
    DiscountCurve(DiscountCurve&&) noexcept = default;
    DiscountCurve& operator=(DiscountCurve&&) noexcept = default;

    // Insert (or overwrite) a node carrying a discount factor at time t.
    void addNode(double t, double df) {
        if (df <= 0.0)
            throw std::domain_error("Discount factor must be strictly positive");
        const double logDf = std::log(df);
        auto it = std::lower_bound(
            nodes_.begin(), nodes_.end(), t,
            [](const CurveNode& nd, double key) { return nd.t < key; });
        if (it != nodes_.end() && std::abs(it->t - t) < 1e-9)
            it->logDf = logDf;
        else
            nodes_.insert(it, CurveNode{t, logDf});
    }

    // Discount factor at time t (days). DF(0)=1 by definition.
    double df(double t) const {
        if (t <= 0.0) return 1.0;
        return std::exp(interp_->interpolateLogDf(nodes_, t));
    }

    // Simple forward rate over [t1,t2]: (DF(t1)/DF(t2)-1)/DCF(t2,t1).
    double forwardRate(double t1, double t2) const {
        const double dcf = dayCountFraction(t2, t1);
        if (std::abs(dcf) < 1e-15)
            throw std::domain_error("Zero day-count fraction in forward rate");
        return (df(t1) / df(t2) - 1.0) / dcf;
    }

    // Exact weights d ln DF(t)/d y_k for every node. DF(0)=1 => all zero.
    std::vector<double> logDfWeights(double t) const {
        if (t <= 0.0) return std::vector<double>(nodes_.size(), 0.0);
        return interp_->logDfWeights(nodes_, t);
    }

    std::vector<double> nodeTimes() const {
        std::vector<double> ts;
        ts.reserve(nodes_.size());
        for (const auto& nd : nodes_) ts.push_back(nd.t);
        return ts;
    }
    const std::vector<CurveNode>& nodes() const { return nodes_; }
    const Interpolator& interpolator() const { return *interp_; }

private:
    std::vector<CurveNode> nodes_;
    std::unique_ptr<Interpolator> interp_;
};

// Instrument abstractions (Q3 extensibility core)

// A priceable instrument: it values itself on a curve and reports the analytic
// sensitivity of its PV to each calibrated node's log discount factor.
// (Implement this interface to add a new product to the pricing/risk system.)
class PricingInstrument {
public:
    virtual ~PricingInstrument() = default;
    virtual double presentValue(const DiscountCurve& curve) const = 0;
    virtual std::vector<double> pvLogDfSensitivity(
        const DiscountCurve& curve) const = 0;
};

// A quoted instrument used to bootstrap the curve. It reports its maturity
// (the node it pins), its observed market quote, the model-implied quote on a
// curve, and the analytic sensitivity of that model quote to node log-DFs.
// (Implement this interface to calibrate the curve from a new quote type.)
class CalibrationInstrument {
public:
    virtual ~CalibrationInstrument() = default;
    virtual double maturityDays() const = 0;
    virtual double marketQuote() const = 0;
    virtual double modelQuote(const DiscountCurve& curve) const = 0;
    virtual std::vector<double> modelQuoteLogDfSensitivity(
        const DiscountCurve& curve) const = 0;
    virtual std::string description() const = 0;
};

// Cash deposit: lend N today, receive N*(1 + r*DCF(T,0)) at T.
//   Market quote  : the cash rate r.
//   Model quote   : r_model = (1/DF(T) - 1) / DCF(T,0).
//   Calibrated DF : 1/(1 + r*DCF(T,0))  (the bootstrapper finds this).
// The model quote depends only on DF(T), so the calibration Jacobian row is
// diagonal, which the generic risk engine handles automatically.
class CashDeposit final : public CalibrationInstrument {
public:
    CashDeposit(double maturityDays, double cashRate, std::string label)
        : t_(maturityDays), rate_(cashRate), label_(std::move(label)) {}

    double maturityDays() const override { return t_; }
    double marketQuote() const override { return rate_; }

    double modelQuote(const DiscountCurve& curve) const override {
        const double delta = dayCountFraction(t_, 0.0);
        return (1.0 / curve.df(t_) - 1.0) / delta;
    }

    std::vector<double> modelQuoteLogDfSensitivity(
        const DiscountCurve& curve) const override {
        // r_model = (e^{-y(T)} - 1)/delta  =>  d/d y(T) = -e^{-y(T)}/delta
        //                                              = -(1/DF(T))/delta
        const double delta = dayCountFraction(t_, 0.0);
        const double df = curve.df(t_);
        const double scale = -(1.0 / df) / delta;
        std::vector<double> w = curve.logDfWeights(t_);
        for (double& wk : w) wk *= scale;
        return w;
    }

    std::string description() const override { return "CashDeposit(" + label_ + ")"; }

private:
    double t_;
    double rate_;
    std::string label_;
};

// Par swap (calibration instrument), fixed & floating paid semi-annually.
//   Par rate P = (1 - DF(T)) / sum_d DCF(d,d_prev)*DF(d)   over the schedule.
//   For maturities <= 6M the swap is a single exchange at maturity, so
//   P collapses to the simple cash form (consistent with the spec note).
//   Market quote : observed par swap rate.
//   Model quote  : P computed from the curve.
class ParSwap final : public CalibrationInstrument {
public:
    static constexpr double kSemiAnnual = 180.0;

    ParSwap(double maturityDays, double parRate, std::string label)
        : t_(maturityDays), rate_(parRate), label_(std::move(label)) {}

    double maturityDays() const override { return t_; }
    double marketQuote() const override { return rate_; }

    double modelQuote(const DiscountCurve& curve) const override {
        const auto dates = schedule();
        double annuity = 0.0, prev = 0.0;
        for (double d : dates) { annuity += dayCountFraction(d, prev) * curve.df(d); prev = d; }
        if (annuity <= 0.0) throw std::runtime_error("Degenerate annuity (ParSwap)");
        return (1.0 - curve.df(t_)) / annuity;
    }

    std::vector<double> modelQuoteLogDfSensitivity(
        const DiscountCurve& curve) const override {
        // P = (1-DF(T))/A,  A = sum_d beta_d DF(d)
        // dP/dy_j = [ -DF(T)*W_j(T)*A - (1-DF(T))*sum_d beta_d DF(d) W_j(d) ] / A^2
        const auto dates = schedule();
        const std::size_t n = curve.nodes().size();
        const double dfT = curve.df(t_);
        const std::vector<double> wT = curve.logDfWeights(t_);

        double A = 0.0;
        std::vector<double> dA(n, 0.0);
        double prev = 0.0;
        for (double d : dates) {
            const double beta = dayCountFraction(d, prev);
            const double dfd = curve.df(d);
            A += beta * dfd;
            const std::vector<double> wd = curve.logDfWeights(d);
            for (std::size_t j = 0; j < n; ++j) dA[j] += beta * dfd * wd[j];
            prev = d;
        }
        if (A <= 0.0) throw std::runtime_error("Degenerate annuity (ParSwap)");
        std::vector<double> out(n, 0.0);
        for (std::size_t j = 0; j < n; ++j)
            out[j] = (-dfT * wT[j] * A - (1.0 - dfT) * dA[j]) / (A * A);
        return out;
    }

    std::string description() const override { return "ParSwap(" + label_ + ")"; }

private:
    std::vector<double> schedule() const {
        std::vector<double> dates;
        if (t_ <= kSemiAnnual + 1e-9) { dates.push_back(t_); return dates; }
        double t = kSemiAnnual;
        while (t < t_ - 1e-9) { dates.push_back(t); t += kSemiAnnual; }
        dates.push_back(t_);
        return dates;
    }
    double t_;
    double rate_;
    std::string label_;
};

// Vanilla interest-rate swap to be priced (general leg frequencies).
//
// Fixed leg : pays  N*r_fixed*DCF(t_i,t_{i-1}) at each fixed date.
// Float leg : receives N*F(t_{j-1},t_j)*DCF(t_j,t_{j-1}) (forwards from curve).
// PV is taken from the fixed-rate payer's side:
//     PV = PV(float received) - PV(fixed paid)
// The floating leg telescopes to N*(1-DF(T)) for any float frequency.
class Swap final : public PricingInstrument {
public:
    Swap(double notional, double fixedRate, double maturityDays,
         double fixedPeriodDays, double floatPeriodDays)
        : notional_(notional), fixedRate_(fixedRate), maturity_(maturityDays),
          fixedPeriod_(fixedPeriodDays), floatPeriod_(floatPeriodDays) {}

    static std::vector<double> makeSchedule(double maturity, double period) {
        std::vector<double> dates;
        double t = period;
        while (t < maturity - 1e-9) { dates.push_back(t); t += period; }
        dates.push_back(maturity);
        return dates;
    }

    double fixedAnnuity(const DiscountCurve& curve) const {
        double annuity = 0.0, prev = 0.0;
        for (double d : makeSchedule(maturity_, fixedPeriod_)) {
            annuity += dayCountFraction(d, prev) * curve.df(d);
            prev = d;
        }
        return annuity;
    }
    double fixedLegPv(const DiscountCurve& curve) const {
        return notional_ * fixedRate_ * fixedAnnuity(curve);
    }
    double floatingLegPv(const DiscountCurve& curve) const {
        double pv = 0.0, prev = 0.0;
        for (double d : makeSchedule(maturity_, floatPeriod_)) {
            pv += notional_ * curve.forwardRate(prev, d) *
                  dayCountFraction(d, prev) * curve.df(d);
            prev = d;
        }
        return pv;
    }
    double presentValue(const DiscountCurve& curve) const override {
        return floatingLegPv(curve) - fixedLegPv(curve);
    }
    double parRate(const DiscountCurve& curve) const {
        const double annuity = fixedAnnuity(curve);
        if (annuity <= 0.0) throw std::runtime_error("Degenerate fixed annuity");
        return floatingLegPv(curve) / (notional_ * annuity);
    }

    // Analytic dPV/dy_k. Using PV = N(1-DF(T)) - N*r_fixed*sum alpha_m DF(a_m):
    //   dPV/dy_k = -N*DF(T)*W_k(T) - N*r_fixed*sum_m alpha_m*DF(a_m)*W_k(a_m).
    std::vector<double> pvLogDfSensitivity(
        const DiscountCurve& curve) const override {
        const std::size_t n = curve.nodes().size();
        std::vector<double> dPv(n, 0.0);

        const double dfT = curve.df(maturity_);
        const std::vector<double> wT = curve.logDfWeights(maturity_);
        for (std::size_t k = 0; k < n; ++k) dPv[k] += -notional_ * dfT * wT[k];

        double prev = 0.0;
        for (double d : makeSchedule(maturity_, fixedPeriod_)) {
            const double alpha = dayCountFraction(d, prev);
            const double dfd = curve.df(d);
            const std::vector<double> wd = curve.logDfWeights(d);
            for (std::size_t k = 0; k < n; ++k)
                dPv[k] += -notional_ * fixedRate_ * alpha * dfd * wd[k];
            prev = d;
        }
        return dPv;
    }

private:
    double notional_, fixedRate_, maturity_, fixedPeriod_, floatPeriod_;
};

// Generic curve calibrator
//
// Bootstraps a curve from any ordered set of CalibrationInstruments. For each
// instrument (ascending maturity) it solves for the node discount factor that
// reprices the instrument's market quote, using the curve-so-far for any
// intermediate dates. The model quote is monotone in the node DF for the
// instruments here, so a bracketed bisection is robust and derivative-free.
class CurveCalibrator {
public:
    using InstrumentPtr = std::unique_ptr<CalibrationInstrument>;

    static DiscountCurve calibrate(const std::vector<InstrumentPtr>& instruments,
                                   std::unique_ptr<Interpolator> interp) {
        DiscountCurve curve(std::move(interp));

        // Calibrate in ascending maturity order.
        std::vector<const CalibrationInstrument*> ordered;
        ordered.reserve(instruments.size());
        for (const auto& inst : instruments) ordered.push_back(inst.get());
        std::sort(ordered.begin(), ordered.end(),
                  [](const CalibrationInstrument* a, const CalibrationInstrument* b) {
                      return a->maturityDays() < b->maturityDays();
                  });

        for (const CalibrationInstrument* inst : ordered) {
            const double t = inst->maturityDays();
            const double df = solveNode(curve, *inst, t);
            curve.addNode(t, df);
        }
        return curve;
    }

private:
    // Solve DF(t) so that modelQuote(curve + trial node) == marketQuote.
    static double solveNode(const DiscountCurve& base,
                            const CalibrationInstrument& inst, double t) {
        const double target = inst.marketQuote();
        auto residual = [&](double dfTrial) {
            DiscountCurve trial = base;   // copy (clones interpolator)
            trial.addNode(t, dfTrial);
            return inst.modelQuote(trial) - target;
        };

        double lo = 1e-8;   // tiny DF  -> very high implied rate
        double hi = 1.0;    // DF = 1   -> implied rate ~ 0
        double fLo = residual(lo);
        double fHi = residual(hi);

        // Expand upper bound for quotes implying DF slightly above 1.
        int guard = 0;
        while (fLo * fHi > 0.0 && guard++ < 100) { hi *= 1.5; fHi = residual(hi); }
        if (fLo * fHi > 0.0)
            throw std::runtime_error("Failed to bracket DF for " + inst.description());

        for (int iter = 0; iter < 200; ++iter) {
            const double mid = 0.5 * (lo + hi);
            const double fMid = residual(mid);
            if (std::abs(fMid) < 1e-14 || (hi - lo) < 1e-14) return mid;
            if (fLo * fMid <= 0.0) hi = mid;
            else { lo = mid; fLo = fMid; }
        }
        return 0.5 * (lo + hi);
    }
};

// Generic analytic risk engine
//
// Computes dPV/dq_i (q_i = the market quote at maturity i) for any priced
// instrument against a curve calibrated from any ordered instrument set,
// fully analytically (no bump-and-revalue).
//
// Chain rule:   dPV/dq_i = sum_k (dPV/dy_k) * (dy_k/dq_i)
//   * dPV/dy_k : from PricingInstrument::pvLogDfSensitivity (full curve).
//   * dy_k/dq_i: from differentiating the sequential bootstrap.
//
// The bootstrap pins node k from instrument k using only nodes 0..k (earlier
// nodes are frozen), so the residual R_k = modelQuote_k - q_k depends on
// y_0..y_k and q_k only. That gives a lower-triangular system:
//
//     sum_{j<=k} (dModelQuote_k/dy_j) * (dy_j/dq_i) = delta_{ki}
//
// solved by forward substitution. This matches re-bootstrapping under a quote
// bump, and collapses to a diagonal system for cash instruments.
class RiskEngine {
public:
    using InstrumentPtr = std::unique_ptr<CalibrationInstrument>;

    static std::vector<double> computeRisk(
        const DiscountCurve& curve,
        const std::vector<InstrumentPtr>& calibrationInstruments,
        const PricingInstrument& priced) {

        // PV sensitivity to each node log-DF (full curve).
        const std::vector<double> g = priced.pvLogDfSensitivity(curve);
        const std::vector<CurveNode>& nodes = curve.nodes();
        const std::size_t n = nodes.size();

        // Calibration instruments in node (ascending maturity) order.
        std::vector<const CalibrationInstrument*> ordered;
        ordered.reserve(calibrationInstruments.size());
        for (const auto& inst : calibrationInstruments) ordered.push_back(inst.get());
        std::sort(ordered.begin(), ordered.end(),
                  [](const CalibrationInstrument* a, const CalibrationInstrument* b) {
                      return a->maturityDays() < b->maturityDays();
                  });
        if (ordered.size() != n)
            throw std::runtime_error("Instrument/node count mismatch in RiskEngine");

        // dydq[k] holds dy_k/dq_i for i = 0..k (lower triangular).
        std::vector<std::vector<double>> dydq(n);

        for (std::size_t k = 0; k < n; ++k) {
            // Sub-curve as it existed when node k was calibrated: nodes 0..k.
            DiscountCurve sub(curve.interpolator().clone());
            for (std::size_t j = 0; j <= k; ++j)
                sub.addNode(nodes[j].t, std::exp(nodes[j].logDf));

            // Row k of the calibration Jacobian: dModelQuote_k/dy_j, j=0..k.
            const std::vector<double> row =
                ordered[k]->modelQuoteLogDfSensitivity(sub);
            if (k >= row.size() || std::abs(row[k]) < 1e-300)
                throw std::runtime_error("Singular calibration Jacobian");

            dydq[k].assign(k + 1, 0.0);
            for (std::size_t i = 0; i <= k; ++i) {
                double rhs = (i == k) ? 1.0 : 0.0;
                for (std::size_t j = i; j < k; ++j) rhs -= row[j] * dydq[j][i];
                dydq[k][i] = rhs / row[k];
            }
        }

        // risk_i = sum_{k>=i} dPV/dy_k * dy_k/dq_i.
        std::vector<double> risk(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = i; k < n; ++k) risk[i] += g[k] * dydq[k][i];
        return risk;
    }
};

// Input parser
struct MarketQuote {
    std::string tenor;
    double days;
    double cashRate;  // decimal
    double parRate;   // decimal
};

struct MarketData {
    std::vector<MarketQuote> quotes;
    double evaluationTime = 0.0;       // t (days) for Q1
    double newSwapNotional = 100.0;
    double newSwapFixedRate = 0.0;     // decimal
    double newSwapMaturity = 0.0;      // days
    double newSwapFixedPeriod = 0.0;   // days
    double newSwapFloatPeriod = 0.0;   // days
};

class InputParser {
public:
    static MarketData parse(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Could not open input file: " + path);

        MarketData data;
        std::string line;

        if (!std::getline(in, line)) throw std::runtime_error("Input file is empty");
        const int count = std::stoi(trim(line));
        if (count <= 0) throw std::runtime_error("Invalid maturity count");

        for (int i = 0; i < count; ++i) {
            if (!std::getline(in, line))
                throw std::runtime_error("Unexpected EOF reading quotes");
            const auto cols = split(line, ',');
            if (cols.size() < 3) throw std::runtime_error("Malformed quote row: " + line);
            MarketQuote q;
            q.tenor = trim(cols[0]);
            q.days = tenorToDays(q.tenor);
            q.cashRate = std::stod(trim(cols[1])) / 100.0;
            q.parRate = std::stod(trim(cols[2])) / 100.0;
            data.quotes.push_back(q);
        }

        if (!std::getline(in, line) || trim(line).empty())
            throw std::runtime_error("Missing evaluation time row");
        data.evaluationTime = std::stod(trim(split(line, ',').front()));

        if (std::getline(in, line) && !trim(line).empty()) {
            const auto cols = split(line, ',');
            if (cols.size() < 4) throw std::runtime_error("Malformed new-swap row: " + line);
            data.newSwapFixedRate = std::stod(trim(cols[0])) / 100.0;
            data.newSwapMaturity = tenorToDays(trim(cols[1]));
            data.newSwapFixedPeriod = frequencyToDays(trim(cols[2]));
            data.newSwapFloatPeriod = frequencyToDays(trim(cols[3]));
        }
        return data;
    }

private:
    static std::string trim(const std::string& s) {
        const auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        const auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }
    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) out.push_back(item);
        return out;
    }
};

// Build calibration instrument sets from the market quotes.
inline std::vector<std::unique_ptr<CalibrationInstrument>> makeCashInstruments(
    const std::vector<MarketQuote>& quotes) {
    std::vector<std::unique_ptr<CalibrationInstrument>> insts;
    for (const auto& q : quotes)
        insts.push_back(std::make_unique<CashDeposit>(q.days, q.cashRate, q.tenor));
    return insts;
}
inline std::vector<std::unique_ptr<CalibrationInstrument>> makeSwapInstruments(
    const std::vector<MarketQuote>& quotes) {
    std::vector<std::unique_ptr<CalibrationInstrument>> insts;
    for (const auto& q : quotes)
        insts.push_back(std::make_unique<ParSwap>(q.days, q.parRate, q.tenor));
    return insts;
}

}  // namespace curve

// Program entry point
int main(int argc, char** argv) {
    using namespace curve;
    try {
        const std::string inputPath = (argc > 1) ? argv[1] : "input.csv";
        const std::string outputPath = (argc > 2) ? argv[2] : "Output.csv";

        const MarketData data = InputParser::parse(inputPath);
        const double t = data.evaluationTime;

        // Calibration instrument sets (one per instrument family).
        const auto cashInsts = makeCashInstruments(data.quotes);
        const auto swapInsts = makeSwapInstruments(data.quotes);

        // Build the four curves through the generic calibrator. Swapping in a
        // different interpolation is a single factory argument (see CubicSpline).
        const DiscountCurve cashLinear = CurveCalibrator::calibrate(
            cashInsts, InterpolatorFactory::create(InterpolationMethod::Linear));
        const DiscountCurve cashAQ = CurveCalibrator::calibrate(
            cashInsts, InterpolatorFactory::create(InterpolationMethod::AveragedQuadratic));
        const DiscountCurve swapLinear = CurveCalibrator::calibrate(
            swapInsts, InterpolatorFactory::create(InterpolationMethod::Linear));
        const DiscountCurve swapAQ = CurveCalibrator::calibrate(
            swapInsts, InterpolatorFactory::create(InterpolationMethod::AveragedQuadratic));

        // Q1: discount factor at t.
        const double q1a = cashLinear.df(t);
        const double q1b = cashAQ.df(t);
        const double q1c = swapLinear.df(t);
        const double q1d = swapAQ.df(t);

        // Q2.1: price the new swap (fixed-rate payer).
        const Swap newSwap(data.newSwapNotional, data.newSwapFixedRate,
                           data.newSwapMaturity, data.newSwapFixedPeriod,
                           data.newSwapFloatPeriod);

        const double pvCashLin = newSwap.presentValue(cashLinear);
        const double pvCashAQ = newSwap.presentValue(cashAQ);
        const double pvSwapLin = newSwap.presentValue(swapLinear);
        const double pvSwapAQ = newSwap.presentValue(swapAQ);

        const double parCashLin = newSwap.parRate(cashLinear);
        const double parCashAQ = newSwap.parRate(cashAQ);
        const double parSwapLin = newSwap.parRate(swapLinear);
        const double parSwapAQ = newSwap.parRate(swapAQ);

        // Q2.2: analytic risk vectors dPV/dq_i.
        const std::vector<double> riskCashLin =
            RiskEngine::computeRisk(cashLinear, cashInsts, newSwap);
        const std::vector<double> riskCashAQ =
            RiskEngine::computeRisk(cashAQ, cashInsts, newSwap);
        const std::vector<double> riskSwapLin =
            RiskEngine::computeRisk(swapLinear, swapInsts, newSwap);
        const std::vector<double> riskSwapAQ =
            RiskEngine::computeRisk(swapAQ, swapInsts, newSwap);

        // Write Output.csv in the required layout.
        // Flip any -0.0 (from tiny risk values) to +0.0 so the printed output
        // stays clean and easy to diff.
        auto fmt = [](double v) { return (v == 0.0) ? 0.0 : v; };

        std::ofstream out(outputPath);
        if (!out) throw std::runtime_error("Could not open output file: " + outputPath);
        out << std::fixed << std::setprecision(10);
        out << fmt(q1a) << ',' << fmt(q1b) << ',' << fmt(q1c) << ',' << fmt(q1d) << '\n';
        out << fmt(pvCashLin) << ',' << fmt(pvCashAQ) << ',' << fmt(pvSwapLin)
            << ',' << fmt(pvSwapAQ) << '\n';
        out << fmt(parCashLin) << ',' << fmt(parCashAQ) << ',' << fmt(parSwapLin)
            << ',' << fmt(parSwapAQ) << '\n';
        const std::size_t nMat = data.quotes.size();
        for (std::size_t i = 0; i < nMat; ++i)
            out << fmt(riskCashLin[i]) << ',' << fmt(riskCashAQ[i]) << ','
                << fmt(riskSwapLin[i]) << ',' << fmt(riskSwapAQ[i]) << '\n';

        // Console echo.
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Q1 (DF at t = " << t << " days)\n";
        std::cout << "  a) Cash / Linear              : " << q1a << '\n';
        std::cout << "  b) Cash / Averaged-Quadratic  : " << q1b << '\n';
        std::cout << "  c) Swap / Linear              : " << q1c << '\n';
        std::cout << "  d) Swap / Averaged-Quadratic  : " << q1d << '\n';
        std::cout << "Q2.1 Present Value (fixed-rate payer)\n";
        std::cout << "  a) Cash / Linear              : " << pvCashLin << '\n';
        std::cout << "  b) Cash / Averaged-Quadratic  : " << pvCashAQ << '\n';
        std::cout << "  c) Swap / Linear              : " << pvSwapLin << '\n';
        std::cout << "  d) Swap / Averaged-Quadratic  : " << pvSwapAQ << '\n';
        std::cout << "Q2.1 Par-Swap Rate\n";
        std::cout << "  a) Cash / Linear              : " << parCashLin << '\n';
        std::cout << "  b) Cash / Averaged-Quadratic  : " << parCashAQ << '\n';
        std::cout << "  c) Swap / Linear              : " << parSwapLin << '\n';
        std::cout << "  d) Swap / Averaged-Quadratic  : " << parSwapAQ << '\n';
        std::cout << "Q2.2 Risk dPV/dq_i (CashLin, CashAQ, SwapLin, SwapAQ)\n";
        for (std::size_t i = 0; i < nMat; ++i)
            std::cout << "  " << std::setw(3) << data.quotes[i].tenor << " : "
                      << std::setw(15) << riskCashLin[i] << ' '
                      << std::setw(15) << riskCashAQ[i] << ' '
                      << std::setw(15) << riskSwapLin[i] << ' '
                      << std::setw(15) << riskSwapAQ[i] << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
