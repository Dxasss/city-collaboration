load('../../new_df/cleaned_data.mat');
fit_params = readtable('../../adapted_Tt6000.csv');

Tt = 6000; Tp_max = 7295;
t_pred = (Tt:Tp_max)';

B    = 200;          % bootstrap 次数
step = 10;           % 训练期降采样步长
q_lo = 0.025; q_hi = 0.975;
band_scale = 1.5;   

cities = {'China_Wuhan', 'Poland_Warsaw', 'Spain_Almeria', ...
          'United Kingdom_London', 'United States_New York City'};%%在此处修改城市
labels = {'Wuhan', 'Warsaw', 'Almeria', 'London', 'New York'};
colors = [0.85 0.33 0.10; 0.47 0.67 0.19; 0.93 0.69 0.13; 0.00 0.45 0.74; 0.49 0.18 0.56];

lb  = [log(1e-8); log(1e-8); log(0.02)];
ub  = [log(2);    log(10);   log(5)];
opts = optimoptions('lsqnonlin','Display','off','MaxIterations',200);

% --- 校准误差尺度（同 fig4a）---
CAL_TT      = [500, 1000, 1500, 2000, 2500, 3000];
CAL_STRIDE  = 25;
CAL_NBINS   = 24;
CAL_DMAX    = 2.1;
CAL_KEDGES  = [1, 10, 30, 100, 400, Inf];
CAL_R2MIN   = 0;
CAL_MIN_CELL = 300;
CAL_MIN_SIDE = 80;
[coef, ~, sup_d, sup_logk] = calibrate_error_scale(numData, cityNames, ...
    CAL_TT, Tt, CAL_STRIDE, CAL_NBINS, CAL_DMAX, CAL_KEDGES, CAL_R2MIN, ...
    CAL_MIN_CELL, CAL_MIN_SIDE);

figure('Position', [100 100 1000 650]); hold on;
h_leg = gobjects(length(cities), 1);

for c = 1:length(cities)
    name = cities{c};
    id_data = find(cityNames == name, 1);
    id_fit  = find(strcmp(string(fit_params.City), name), 1);
    if isempty(id_data) || isempty(id_fit), continue; end

    data = numData(id_data, :);
    eta_i   = fit_params.eta(id_fit);
    beta_i  = fit_params.beta(id_fit);
    alpha_i = fit_params.alpha(id_fit);
    if ~isfinite(eta_i) || ~isfinite(beta_i) || ~isfinite(alpha_i), continue; end
    if data(Tt) < 1, continue; end

    t0  = find(data > 0, 1, 'first');
    tau = (1:(Tt - t0 + 1))';
    kt  = data(Tt);
    c_obs = smooth_cum(data(t0:Tt));
    th0 = [log(eta_i); log(beta_i); log(alpha_i)];

    idx_ds = 1:step:numel(tau);
    tau_ds   = tau(idx_ds);
    c_obs_ds = c_obs(idx_ds);

    cmod = @(th) (1 + exp(th(2))*tau_ds.^exp(th(3))).^(exp(th(1))/exp(th(2))) - 1;
    logk = @(th) log(kt) + (exp(th(1))/exp(th(2))) * ( ...
        log1p(exp(th(2))*(t_pred - t0 + 1).^exp(th(3))) - ...
        log1p(exp(th(2))*(Tt - t0 + 1).^exp(th(3))) );

    c_hat = cmod(th0);
    resid = c_obs_ds - c_hat;
    n = numel(resid);

    k_star = exp(logk(th0));          % 点预测

    delta = max(log(max(t_pred ./ Tt, 1)), 0);
    s_vec = scale_at(coef, delta, kt, sup_d, sup_logk);
    band_lo = k_star .* exp(-band_scale * s_vec);
    band_hi = k_star .* exp(band_scale * s_vec);

    curves_par = zeros(numel(t_pred), B);
    for b = 1:B
        c_star  = c_hat + resid(randi(n, n, 1));
        th_star = lsqnonlin(@(th) c_star - cmod(th), th0, lb, ub, opts);
        curves_par(:, b) = exp(logk(th_star));
    end
    par_lo = quantile(curves_par, q_lo, 2);
    par_hi = quantile(curves_par, q_hi, 2);

    fill([t_pred; flipud(t_pred)], [band_hi; flipud(band_lo)], ...
         colors(c, :), 'EdgeColor', 'none', 'FaceAlpha', 0.12, 'HandleVisibility', 'off');
    plot(1:Tp_max, data(1:Tp_max), '-', 'Color', [colors(c,:) 0.35], ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    h_leg(c) = plot(t_pred, k_star, '-', 'Color', colors(c, :), 'LineWidth', 2.5, ...
        'DisplayName', labels{c});
end

xline(Tt, 'k--', 'LineWidth', 1.5);
xlabel('t (days)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative collaborations', 'FontSize', 14, 'FontWeight', 'bold');
legend(h_leg(isgraphics(h_leg)), 'Location', 'northwest', 'FontSize', 10);
grid on; set(gca, 'FontSize', 13, 'LineWidth', 1);

savefig(gcf, 'fig4a_bootstrap.fig');
print(gcf, 'fig4a_bootstrap_preview.png', '-dpng', '-r200');
fprintf('Saved fig4a_bootstrap.fig / fig4a_bootstrap_preview.png\n');

% ---------- 局部函数 ----------
function cs = smooth_cum(ci)
    ci = double(ci(:));
    cs = ci;
    jt = []; jv = [];
    for t = 1:length(ci)
        pr = 0; if t > 1, pr = ci(t-1); end
        if ci(t) > pr, jt(end+1) = t; jv(end+1) = ci(t); end
    end
    for k = 1:length(jt)-1
        a0 = jt(k); a1 = jt(k+1);
        if a1 > a0
            idx = a0:(a1-1);
            cs(idx) = jv(k) + (jv(k+1) - jv(k)) * (idx' - a0) / (a1 - a0);
        end
    end
end

function s = scale_at(coef, delta, kt, sup_d, sup_logk)
    delta = delta(:);
    s = zeros(size(delta));
    m = delta > 0;
    if ~any(m), return; end
    kt_use = repmat(max(kt, 1), size(delta));
    if ~isempty(sup_d)
        kt_sup = exp(interp1(sup_d, sup_logk, delta, 'linear', NaN));
        kt_sup(delta < sup_d(1))   = exp(sup_logk(1));
        kt_sup(delta > sup_d(end)) = exp(sup_logk(end));
        kt_use = min(kt_use, max(kt_sup, 1));
    end
    s(m) = exp(coef(1) + coef(2) * log(delta(m)) + coef(3) * log(kt_use(m)));
end

function [coef, cells, sup_d, sup_logk] = calibrate_error_scale(numData, ...
        cityNames, CAL_TT, Tt, STRIDE, NBINS, DMAX, KEDGES, R2MIN, MIN_CELL, MIN_SIDE)
    cn = string(cityNames);
    Dc = {}; Rc = {}; Kc = {};
    n_drop = 0;
    for k = 1:numel(CAL_TT)
        tt = CAL_TT(k);
        T = readtable(sprintf('../../adapted_Tt%d.csv', tt));
        [tf, loc] = ismember(string(T.City), cn);
        t = ((tt+1):STRIDE:Tt)';
        for i = 1:height(T)
            if ~tf(i), continue; end
            e = T.eta(i); bb = T.beta(i); a = T.alpha(i); r2 = T.R2_train(i);
            if ~isfinite(e) || ~isfinite(bb) || ~isfinite(a) || bb <= 0, continue; end
            if ~isfinite(r2) || r2 <= R2MIN, n_drop = n_drop + 1; continue; end
            data = numData(loc(i), :);
            nz = find(data > 0, 1, 'first');
            if isempty(nz), continue; end
            ktf = data(tt);
            if ktf < 1, continue; end
            tau  = max(t - nz + 1, 1);
            tauT = max(tt - nz + 1, 1);
            pred = ktf * ((1 + bb * tau.^a) ./ (1 + bb * tauT.^a)).^(e / bb);
            act  = data(t)';
            ok = isfinite(pred) & pred > 0 & act > 0;
            if sum(ok) < 5, continue; end
            Dc{end+1} = log(t(ok) / tt);
            Rc{end+1} = log(act(ok)) - log(pred(ok));
            Kc{end+1} = repmat(ktf, sum(ok), 1);
        end
    end
    D = vertcat(Dc{:}); Rr = vertcat(Rc{:}); K = vertcat(Kc{:});

    edges  = linspace(0, min(max(D), DMAX), NBINS + 1);
    cells  = [];
    for b = 1:NBINS
        for q = 1:(numel(KEDGES) - 1)
            m = D >= edges(b) & D < edges(b+1) & K > KEDGES(q) & K <= KEDGES(q+1);
            if sum(m) < MIN_CELL, continue; end
            rr = Rr(m);
            if sum(rr > 0) < MIN_SIDE || sum(rr < 0) < MIN_SIDE, continue; end
            cells(end+1, :) = [0.5*(edges(b)+edges(b+1)), exp(mean(log(K(m)))), ...
                sqrt(mean(rr.^2)), sum(m)];
        end
    end
    if size(cells, 1) < 8, error('too few calibration cells'); end

    X = [ones(size(cells,1),1), log(cells(:,1)), log(cells(:,2))];
    w = sqrt(cells(:,4));
    y = log(cells(:,3));
    coef = (X .* w) \ (y .* w);

    [dv, ord] = sort(cells(:,1));
    kv  = cells(ord, 2);
    sup = flipud(cummax(flipud(kv)));
    [sup_d, ~, grp] = unique(dv);
    sup_max  = accumarray(grp, sup, [], @max);
    sup_logk = log(max(flipud(cummax(flipud(sup_max))), 1));
end
