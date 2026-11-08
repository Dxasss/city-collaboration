k_time_df=readtable('../../cleaned_data_2686_cities.csv','NumHeaderLines', 1);
varTypes = varfun(@class, k_time_df, 'OutputFormat', 'cell');
isNumeric = cellfun(@(c) strcmp(c,'double')||strcmp(c,'single'), varTypes);
numData = table2array(k_time_df(:, isNumeric)); numData(isnan(numData)) = 0;
cityNames = string(k_time_df.Var1);
finalValue = numData(:, end); changeCount = sum(diff(numData,1,2)~=0, 2);
iddx = finalValue>=5 & changeCount>=5;
numData = numData(iddx,:); cityNames = cityNames(iddx);
full_params = readtable('../../cities_params.csv');
fprintf('Loaded: %d cities\n', size(numData,1));

Tp_max = 7305; m_val = 1;
log_beta_lb = -4; lambda_beta = 1e4;
lb = [log(1e-8); log(1e-8); log(0.02)]; ub = [log(2); log(10); log(5)];
fmin_opts = optimset('Display','off','Algorithm','sqp','MaxIter',800,'TolX',1e-10,'TolFun',1e-10);

Tt_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000];

for t_idx = 1:length(Tt_list)
    Tt = Tt_list(t_idx);
    lambda_self = 10^(3 + (Tt - 500)/(7000 - 500) * 4);
    fprintf('\n===== Tt=%d (lambda=%.2e) =====\n', Tt, lambda_self);

    train_data = numData(:,1:Tt);
    N = size(train_data,1);
    city_info = build_all(train_data, N, Tt);
    valid_idx = find(~cellfun(@isempty, city_info));
    N_v = length(valid_idx);

    tic;
    eta_f = zeros(N_v,1); beta_f = zeros(N_v,1); alpha_f = zeros(N_v,1);
    gamma_f = zeros(N_v,1); r2_f = zeros(N_v,1);

    for ki = 1:N_v
        info = city_info{valid_idx(ki)};
        tau = info.tau; c_obs = info.c_sm;
        city_name = string(cityNames(valid_idx(ki)));
        j = find(strcmp(string(full_params.City), city_name), 1);
        if ~isempty(j)
            le0 = log(full_params.eta(j)); lb0 = log(max(full_params.beta(j),1e-12)); la0 = log(full_params.alpha(j));
        else
            le0=-1; lb0=-3; la0=0;
        end
        try
            obj = @(x) sse_self(x(1),x(2),x(3), tau, c_obs, le0,lb0,la0, lambda_self, lambda_beta, log_beta_lb);
            [xo,~] = fmincon(obj, [le0;lb0;la0], [],[],[],[], lb, ub, [], fmin_opts);
            eta_f(ki)=exp(xo(1)); beta_f(ki)=exp(xo(2)); alpha_f(ki)=exp(xo(3));
        catch
            eta_f(ki)=exp(le0); beta_f(ki)=exp(lb0); alpha_f(ki)=exp(la0);
        end
        g = eta_f(ki)/beta_f(ki);
        gamma_f(ki) = g;
        cp = m_val*(1+beta_f(ki)*tau.^alpha_f(ki)).^g - m_val;
        r2_f(ki) = r2_score(c_obs, cp);
        if mod(ki,500)==0, e=toc; fprintf('  %d/%d ETA=%.0fm\n', ki, N_v, e/ki*(N_v-ki)/60); end
    end
    fprintf('Fit done: %.1f min\n', toc/60);

    t_pred = (Tt:Tp_max)';
    mape_v = zeros(N,1); ad_v = zeros(N,1);
    for i = 1:N
        data = numData(i,:);
        ad_v(i) = sum(diff(data(1:Tt)) > 0);
        vi = find(valid_idx==i,1);
        if isempty(vi), mape_v(i)=NaN; continue; end
        ei=eta_f(vi); bi=beta_f(vi); ai=alpha_f(vi); g=ei/bi;
        t0=find(data>0,1,'first'); if isempty(t0), continue; end
        tp=max(t_pred-t0+1,1); tT=max(Tt-t0+1,1);
        kt=data(Tt);
        if kt<1
            tt=(Tt+1):Tp_max; act=data(tt); v=act>0;
            if sum(v)>=1, mape_v(i)=mean(abs(act(v))./act(v))*100; end
            continue;
        end
        rt=(1+bi*tp.^ai)./(1+bi*tT.^ai); ks=kt.*rt.^g;
        tt=(Tt+1):Tp_max; act=data(tt); prd=interp1(t_pred,ks,tt,'linear');
        v=act>0;
        if sum(v)>=1, mape_v(i)=mean(abs(act(v)-prd(v))./act(v))*100; end
    end
    fprintf('MAPE: med=%.2f%%, valid=%d/%d\n', median(mape_v,'omitnan'), sum(isfinite(mape_v)), N);

    eta_out=NaN(N,1); beta_out=NaN(N,1); alpha_out=NaN(N,1); gamma_out=NaN(N,1); r2_out=NaN(N,1);
    for ki=1:N_v
        vi=valid_idx(ki);
        eta_out(vi)=eta_f(ki); beta_out(vi)=beta_f(ki); alpha_out(vi)=alpha_f(ki);
        gamma_out(vi)=gamma_f(ki); r2_out(vi)=r2_f(ki);
    end
    T_out = table(string(cityNames(:)), eta_out(:), beta_out(:), alpha_out(:), gamma_out(:), r2_out(:), mape_v(:), ad_v(:), ...
        'VariableNames', {'City','eta','beta','alpha','gamma','R2_train','MAPE','ActiveDays'});
    fn = sprintf('adapt_Tt%d.csv', Tt);
    writetable(T_out, fn);
    fprintf('Saved %s\n', fn);
end
fprintf('\n===== DONE =====\n');

function sse = sse_self(le,lb,la,tau,co,le_full,lb_full,la_full,lam,lbeta,lbl)
    e=exp(le); b=exp(lb); a=exp(la);
    if e<=0||b<=0||a<=0||~isfinite(e)||~isfinite(b)||~isfinite(a), sse=1e15; return; end
    g=e/b; if g>1e8, sse=1e15; return; end
    cp=(1+b*tau.^a).^g-1;
    if any(~isfinite(cp)|cp<0), sse=1e15; return; end
    sse=sum((co-cp).^2);
    if ~isfinite(sse), sse=1e15; return; end
    sse=sse+lam*((le-le_full)^2+(lb-lb_full)^2+(la-la_full)^2);
    if lbeta>0, sse=sse+lbeta*max(0,lbl-lb)^2; end
end

function ic=build_all(td,N,te)
    ic=cell(N,1);
    for i=1:N
        ci=td(i,:)'; nz=find(ci>0,1,'first');
        if isempty(nz), ic{i}=struct('t0',1,'T_eff',te,'M',0,'c_sm',zeros(te,1),'tau',(1:te)');
        else, ic{i}=struct('t0',nz,'T_eff',te-nz+1,'M',ci(end),'c_sm',sc(ci(nz:end)),'tau',(1:te-nz+1)'); end
    end
end
function cs=sc(ci)
    cs=double(ci(:)); jt=0; jv=0;
    for t=1:length(ci), pr=0; if t>1, pr=ci(t-1); end; if ci(t)>pr, jt(end+1)=t; jv(end+1)=double(ci(t)); end; end
    for kk=1:length(jt)-1, for t=max(jt(kk),1):jt(kk+1), cs(t)=jv(kk)+(jv(kk+1)-jv(kk))*(t-jt(kk))/(jt(kk+1)-jt(kk)); end; end
end
function r2=r2_score(o,p), sr=sum((o-p).^2); st=sum((o-mean(o)).^2); if st<1e-10, r2=0; else, r2=1-sr/st; end; end
