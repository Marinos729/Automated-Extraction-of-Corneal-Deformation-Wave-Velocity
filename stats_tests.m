% ============================================================================
%  stats_tests.m — statistical analysis companion to wave_pipeline_unified.m
% ----------------------------------------------------------------------------
%  Reads all_results.csv (written by the pipeline) and reproduces the thesis
%  Results:
%     * descriptive statistics of the propagation velocity (m/s);
%     * ONE test per comparison, chosen by that comparison's own distribution:
%         - age      : independent two-sample t-test (pooled variance) on the
%                      eye-averaged velocity. The equal-variance assumption is
%                      checked with an F-test and each group's normality with
%                      the Shapiro-Wilk test.
%         - meridian : Wilcoxon signed-rank test on the within-eye
%                      nasal-minus-temporal differences (the differences are
%                      non-normal), summarised by the median difference with a
%                      distribution-free Hodges-Lehmann 95% confidence interval.
%  The two comparisons test distinct hypotheses, so no multiplicity adjustment
%  is applied.
%
%  The reported velocity is v_profile (the profile-method slope). The binary
%  method is only an internal verifier and is not used here.
%
%  REQUIRES: Statistics and Machine Learning Toolbox (ttest2, vartest2,
%  signrank, tiedrank, norminv, normcdf). The Shapiro-Wilk test is implemented
%  below (Royston 1992), so no extra download is needed.
% ============================================================================

clear; clc;

% ---- config ---------------------------------------------------------------
csvPath    = 'all_results.csv';     % path to the pipeline output (edit as needed)
mmPerPx    = 8.5 / 576;             % 0.014756944 mm/px  (as in the pipeline)
msPerFrame = 1000 / 4330;           % 0.230946882 ms/frame
k          = mmPerPx / msPerFrame;  % 0.063898 m/s per (arc px/frame)

% ---- load & convert to m/s ------------------------------------------------
T   = readtable(csvPath, 'TextType','string');
vms = T.v_profile * k;                       % native units are arc px/frame
age = string(T.age);
mer = string(T.meridian);
eyeKey = age + "|" + string(T.eye) + "|" + string(T.subject);   % one eye = one triple
fprintf('Loaded %d sequences from %d eyes.\n\n', height(T), numel(unique(eyeKey)));

% ---- descriptive statistics (m/s) -----------------------------------------
fprintf('==== DESCRIPTIVE: propagation velocity (m/s) ====\n');
describe('all',      vms);                     % expect n=101 mean 0.121 sd 0.059 median 0.113
describe('Below_26', vms(age=="Below_26"));    % expect n=56  mean 0.129 sd 0.059 median 0.123
describe('Above_44', vms(age=="Above_44"));    % expect n=45  mean 0.111 sd 0.059 median 0.102
describe('nasal',    vms(mer=="nasal"));       % expect n=50  mean 0.140 sd 0.055 median 0.154
describe('temporal', vms(mer=="temporal"));    % expect n=51  mean 0.102 sd 0.058 median 0.091
fprintf('\n');

% ---- AGE: independent two-sample t-test (eye-averaged, pooled variance) ----
[eyeV, eyeAge] = eyeLevel(vms, eyeKey);   % average the two meridians per eye
a = eyeV(eyeAge=="Below_26");  a = a(~isnan(a));   % younger
b = eyeV(eyeAge=="Above_44");  b = b(~isnan(b));   % older
fprintf('==== AGE  (two-sample t, eye-level, n=%d vs %d) ====\n', numel(a), numel(b));
[Wa, pa] = swtest(a);   [Wb, pb] = swtest(b);
fprintf('  normality (Shapiro-Wilk): Below W=%.3f p=%.3f | Above W=%.3f p=%.3f  [expect ~0.96/0.35, 0.94/0.18]\n', ...
        Wa, pa, Wb, pb);
[~, pF, ~, stF]    = vartest2(a, b);      % F-test: equal-variance assumption of the pooled t
fprintf('  equal variance (F-test): F(%d,%d)=%.3f p=%.3f  [expect F=0.62, p=0.23]\n', ...
        stF.df1, stF.df2, stF.fstat, pF);
[~, pAge, ciAge, stAge] = ttest2(a, b);   % pooled (equal-variance) two-sample t is the default
fprintf('  Below=%.4f  Above=%.4f  diff=%+.4f m/s\n', mean(a), mean(b), mean(a)-mean(b));
fprintf('  t(%d)=%.3f  p=%.4f | 95%% CI=[%+.4f, %+.4f]\n', stAge.df, stAge.tstat, pAge, ciAge(1), ciAge(2));
fprintf('  [expect: diff=+0.018, t(49)=1.53, p=0.13, CI(-0.006,+0.041)]\n\n');

% ---- MERIDIAN: Wilcoxon signed-rank (within-eye pairs, diffs non-normal) ----
[nas, tmp] = pairByEye(vms, eyeKey, mer);   % NaN pairs (e.g. a rejected meridian) dropped inside
d = nas - tmp;
fprintf('==== MERIDIAN  (Wilcoxon signed-rank, %d eye pairs) ====\n', numel(d));
[Wd, pd_] = swtest(d);
fprintf('  normality of differences (Shapiro-Wilk): W=%.3f p=%.4f  [expect 0.94, 0.010 -> non-normal]\n', Wd, pd_);
[pMer, ~, stMer] = signrank(nas, tmp);
zMer = signedRankZ(d);                       % continuity-corrected large-sample z
[HL, loCI, hiCI] = hodgesLehmann(d);         % distribution-free 95% CI on the median shift
fprintf('  nasal=%.4f  temporal=%.4f  median diff=%+.4f m/s  (nasal faster in %d of %d)\n', ...
        mean(nas), mean(tmp), median(d), sum(d>0), numel(d));
fprintf('  Wilcoxon W=%.0f  |z|=%.3f  p=%.5f | Hodges-Lehmann=%+.4f  95%% CI=[%+.4f, %+.4f]\n', ...
        stMer.signedrank, abs(zMer), pMer, HL, loCI, hiCI);
fprintf('  [expect: median=+0.052, |z|=3.66, p<0.001, HL=+0.043, CI(+0.022,+0.063), 36 of 50]\n\n');

% ============================================================================
%  LOCAL FUNCTIONS
% ============================================================================
function describe(label, x)
    x = x(~isnan(x));
    fprintf('  %-9s n=%3d  mean=%.4f  sd=%.4f  median=%.4f\n', ...
        label, numel(x), mean(x), std(x), median(x));
end

function [eyeV, eyeAge] = eyeLevel(v, key)
    [G, gkey] = findgroups(key);
    eyeV   = splitapply(@(z) mean(z,'omitnan'), v, G);
    eyeAge = extractBefore(gkey, "|");
end

function [nas, tmp] = pairByEye(v, key, mer)
    keyN = key(mer=="nasal");    vN = v(mer=="nasal");
    keyT = key(mer=="temporal"); vT = v(mer=="temporal");
    [tf, loc] = ismember(keyN, keyT);
    nas = vN(tf);        nas = nas(:);
    tmp = vT(loc(tf));   tmp = tmp(:);
    ok  = ~isnan(nas) & ~isnan(tmp);          % drop pairs where either meridian was rejected
    nas = nas(ok);  tmp = tmp(ok);
end

function z = signedRankZ(d)
% Continuity-corrected large-sample z for the Wilcoxon signed-rank statistic.
    d = d(~isnan(d));  d = d(d~=0);
    r = tiedrank(abs(d));
    Rp = sum(r(d>0));  Rm = sum(r(d<0));
    Tw = min(Rp, Rm);  n = numel(d);
    mu = n*(n+1)/4;    sd = sqrt(n*(n+1)*(2*n+1)/24);
    z  = (Tw - mu + 0.5) / sd;                % negative when the positive ranks dominate
end

function [HL, lo, hi] = hodgesLehmann(d)
% Hodges-Lehmann median-shift estimate and a distribution-free 95% CI from the
% Walsh averages of the paired differences.
    d = d(~isnan(d));  n = numel(d);
    [I, J] = find(triu(true(n)));             % all i <= j pairs
    walsh  = sort((d(I) + d(J)) / 2);
    M   = numel(walsh);
    HL  = median(walsh);
    muW = n*(n+1)/4;   sdW = sqrt(n*(n+1)*(2*n+1)/24);
    kk  = max(round(muW - 1.96*sdW), 1);
    lo  = walsh(kk);   hi = walsh(M - kk + 1);
end

function [W, pValue] = swtest(x)
% Shapiro-Wilk test of composite normality, Royston (1992) approximation.
% Valid for 4 <= n <= 5000 and matches R's shapiro.test and SciPy's shapiro
% to about three decimals.
%   H0 : x is drawn from a normal distribution.
%   W      : the Shapiro-Wilk statistic.
%   pValue : upper-tail p-value (small p rejects normality).
% Reference: Royston P. (1992), Statistics and Computing 2, 117-119.
    x = sort(x(~isnan(x)));  x = x(:);
    n = numel(x);
    if n < 4, error('swtest: need at least 4 observations.'); end
    i      = (1:n)';
    mtilde = norminv((i - 0.375) / (n + 0.25));
    m2     = mtilde' * mtilde;
    u      = 1 / sqrt(n);
    pn  = [-2.706056 4.434685 -2.071190 -0.147981 0.221157 0];
    pm  = [-3.582633 5.682633 -1.752461 -0.293762 0.042981 0];
    an  = polyval(pn, u) + mtilde(n)   / sqrt(m2);
    an1 = polyval(pm, u) + mtilde(n-1) / sqrt(m2);
    a = zeros(n,1);
    if n <= 5
        phi      = (m2 - 2*mtilde(n)^2) / (1 - 2*an^2);
        a(n)     =  an;   a(1) = -an;
        a(2:n-1) =  mtilde(2:n-1) / sqrt(phi);
    else
        phi      = (m2 - 2*mtilde(n)^2 - 2*mtilde(n-1)^2) / (1 - 2*an^2 - 2*an1^2);
        a(n)     =  an;   a(n-1) =  an1;
        a(1)     = -an;   a(2)   = -an1;
        a(3:n-2) =  mtilde(3:n-2) / sqrt(phi);
    end
    W = (a' * x)^2 / sum((x - mean(x)).^2);
    if n <= 11
        g   = polyval([0.459 -2.273], n);
        mu  = polyval([-6.714e-4 0.025054 -0.39978 0.5440], n);
        sig = exp(polyval([-0.0020322 0.062767 -0.77857 1.3822], n));
        z   = (-log(g - log(1 - W)) - mu) / sig;
    else
        lnn = log(n);
        mu  = polyval([0.0038915 -0.083751 -0.31082 -1.5861], lnn);
        sig = exp(polyval([0.0030302 -0.082676 -0.4803], lnn));
        z   = (log(1 - W) - mu) / sig;
    end
    pValue = 1 - normcdf(z);
end
