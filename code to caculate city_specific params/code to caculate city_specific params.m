load('../../cleaned_data_2686_cities.mat');
varTypes = varfun(@class, k_time_df, 'OutputFormat', 'cell');
isNumeric = cellfun(@(c) strcmp(c,'double')||strcmp(c,'single'), varTypes);
numData = table2array(k_time_df(:, isNumeric)); numData(isnan(numData)) = 0;
cityNames = string(k_time_df.Var1);
finalValue = numData(:, end); changeCount = sum(diff(numData,1,2)~=0, 2);
iddx = finalValue>=5 & changeCount>=5;
numData = numData(iddx,:); cityNames = cityNames(iddx);
N = size(numData,1); train_end = 7305; m = 1;

tau_cell=cell(N,1); cobs_cell=cell(N,1); M_vec=zeros(N,1); valid=false(N,1);
for i=1:N
    ci=numData(i,:)'; nz=find(ci>0,1,'first');
    if isempty(nz)||train_end-nz<10, continue; end
    Mi=ci(end); if Mi<2, continue; end
    c_sm=smooth_cum(ci(nz:end)); tau_cell{i}=(1:numel(c_sm))'; cobs_cell{i}=c_sm(:); M_vec(i)=Mi; valid(i)=true;
end
valid_idx=find(valid); N_v=length(valid_idx);
logM=log(M_vec(valid_idx));
[~,rk]=sort(M_vec(valid_idx)); rank_M=zeros(N_v,1); rank_M(rk)=(1:N_v)';
logM_rank=quantile(logM,(rank_M-0.5)/N_v);
cityNames_valid = cityNames(valid_idx);

log_eta_grid=linspace(-5,2,18); log_beta_grid=linspace(-6,1,18); log_alpha_grid=linspace(log(0.2),log(3.5),12);
eta_g=exp(log_eta_grid); beta_g=exp(log_beta_grid); alpha_g=exp(log_alpha_grid);
log_beta_lb_soft=-4; lambda_beta=1e4;
lb=[log(1e-8);log(1e-8);log(0.02)]; ub=[log(2);log(10);log(5)];
fmin_opts=optimset('Display','off','Algorithm','sqp','MaxIter',800,'TolX',1e-10,'TolFun',1e-10);

eta_r1=zeros(N_v,1); beta_r1=zeros(N_v,1); alpha_r1=zeros(N_v,1); r2_r1=zeros(N_v,1);
kbj = find(cityNames_valid=="China_Beijing",1);
for ki=1:N_v
    tau=tau_cell{valid_idx(ki)}; c_obs=cobs_cell{valid_idx(ki)};
    sse_best=Inf; le0=0; lb0=-3; la0=0;
    for ia=1:12, ta=tau.^alpha_g(ia);
        for ib=1:18, base=1+beta_g(ib)*ta;
            for ie=1:18, gam=eta_g(ie)/beta_g(ib); cp=base.^gam-1;
                if any(~isfinite(cp)), continue; end
                s=sum((c_obs-cp).^2);
                if s<sse_best, sse_best=s; le0=log_eta_grid(ie); lb0=log_beta_grid(ib); la0=log_alpha_grid(ia); end
            end
        end
    end
    if ki==kbj, le0=log(0.3); lb0=-3.25; la0=log(0.42); end  
    x=fmincon(@(x)sse_free(x(1),x(2),x(3),tau,c_obs),[le0;lb0;la0],[],[],[],[],lb,ub,[],fmin_opts);
    eta_r1(ki)=exp(x(1)); beta_r1(ki)=exp(x(2)); alpha_r1(ki)=exp(x(3));
    g=eta_r1(ki)/beta_r1(ki); cp=(1+beta_r1(ki)*tau.^alpha_r1(ki)).^g-1;
    r2_r1(ki)=r2_score(c_obs,cp);
    if mod(ki,300)==0, fprintf('  R1 %d/%d medR2=%.4f t=%.0fs\n',ki,N_v,median(r2_r1(1:ki)),toc(t0)); end
end

good=r2_r1>0.5 & eta_r1>0;
p_rank=polyfit(logM_rank(good),log(eta_r1(good)),1);
slope=p_rank(1); inter=p_rank(2);

lambda_val=1e8; M_thr_big=10000;
eta_f=zeros(N_v,1); beta_f=zeros(N_v,1); alpha_f=zeros(N_v,1); gamma_f=zeros(N_v,1); r2_f=zeros(N_v,1);
for ki=1:N_v
    if M_vec(valid_idx(ki))>=M_thr_big
        eta_f(ki)=eta_r1(ki); beta_f(ki)=beta_r1(ki); alpha_f(ki)=alpha_r1(ki); gamma_f(ki)=eta_r1(ki)/beta_r1(ki); r2_f(ki)=r2_r1(ki); continue;
    end
    tau=tau_cell{valid_idx(ki)}; c_obs=cobs_cell{valid_idx(ki)}; logM_k=logM_rank(ki);
    le0=log(max(eta_r1(ki),1e-6)); lb0=log(max(beta_r1(ki),exp(log_beta_lb_soft))); la0=log(max(alpha_r1(ki),1e-4));
    obj=@(x)sse_prior(x(1),x(2),x(3),tau,c_obs,lambda_val,slope,inter,logM_k,0,log_beta_lb_soft);
    x=fmincon(obj,[le0;lb0;la0],[],[],[],[],lb,ub,[],fmin_opts);
    ei=exp(x(1)); bi=exp(x(2)); ai=exp(x(3)); cp=(1+bi*tau.^ai).^(ei/bi)-1;
    if any(~isfinite(cp)), eta_f(ki)=eta_r1(ki); beta_f(ki)=beta_r1(ki); alpha_f(ki)=alpha_r1(ki); gamma_f(ki)=eta_r1(ki)/beta_r1(ki); r2_f(ki)=r2_r1(ki); continue; end
    eta_f(ki)=ei; beta_f(ki)=bi; alpha_f(ki)=ai; gamma_f(ki)=ei/bi; r2_f(ki)=r2_score(c_obs,cp);
end

eta_pred1 = exp(slope*logM_rank + inter);
eta_dev = abs(log(eta_f) - log(eta_pred1));
step1 = (r2_f>=0.5 & eta_dev>0.5) & (M_vec(valid_idx)<M_thr_big);
lb_relax=[log(1e-8);log(1e-12);log(0.02)]; ub_relax=[log(2);log(10);log(10)];
for ki=1:N_v
    if ~step1(ki), continue; end
    tau=tau_cell{valid_idx(ki)}; c_obs=cobs_cell{valid_idx(ki)}; logM_k=logM_rank(ki);
    le0=log(max(eta_f(ki),1e-6)); lb0=log(max(beta_f(ki),1e-12)); la0=log(max(alpha_f(ki),0.05));
    obj=@(x)sse_prior(x(1),x(2),x(3),tau,c_obs,1e8,slope,inter,logM_k,0,log_beta_lb_soft);
    x=fmincon(obj,[le0;lb0;la0],[],[],[],[],lb_relax,ub_relax,[],fmin_opts);
    ei=exp(x(1)); bi=exp(x(2)); ai=exp(x(3)); cp=(1+bi*tau.^ai).^(ei/bi)-1;
    if any(~isfinite(cp)), continue; end
    eta_f(ki)=ei; beta_f(ki)=bi; alpha_f(ki)=ai; gamma_f(ki)=ei/bi; r2_f(ki)=r2_score(c_obs,cp);
end
kparis = find(cityNames_valid=="France_Paris",1);

step2 = (alpha_f>=4.9 | beta_f<=1e-7 | r2_f<0.5) & ~step1 & (M_vec(valid_idx)<M_thr_big);
fmin_opts3=optimset('Display','off','Algorithm','active-set','MaxIter',1000,'TolX',1e-10,'TolFun',1e-10);
le_g=linspace(-6,1,8); lb_g=linspace(-6,1,8); la_g=linspace(log(0.2),log(10),8);
for ki=1:N_v
    if ~step2(ki), continue; end
    tau=tau_cell{valid_idx(ki)}; c_obs=cobs_cell{valid_idx(ki)}; logM_k=logM_rank(ki);
    le0=log(max(eta_f(ki),1e-6)); lb0=log(max(beta_f(ki),1e-12)); la0=log(max(alpha_f(ki),0.05));
    obj=@(x)sse_prior(x(1),x(2),x(3),tau,c_obs,1e4,slope,inter,logM_k,0,log_beta_lb_soft);
    sbest=obj([le0;lb0;la0]);
    for ig1=1:8, for ig2=1:8, for ig3=1:8
        ss=obj([le_g(ig1);lb_g(ig2);la_g(ig3)]);
        if ss<sbest, sbest=ss; le0=le_g(ig1); lb0=lb_g(ig2); la0=la_g(ig3); end
    end, end, end
    x=fmincon(obj,[le0;lb0;la0],[],[],[],[],lb_relax,ub_relax,[],fmin_opts3);
    ei=exp(x(1)); bi=exp(x(2)); ai=exp(x(3)); cp=(1+bi*tau.^ai).^(ei/bi)-1;
    if any(~isfinite(cp)), continue; end
    eta_f(ki)=ei; beta_f(ki)=bi; alpha_f(ki)=ai; gamma_f(ki)=ei/bi; r2_f(ki)=r2_score(c_obs,cp);
end
T_out=table(cityNames_valid(:), M_vec(valid_idx), eta_f, beta_f, alpha_f, gamma_f, r2_f, ...
    'VariableNames',{'City','M_final','eta','beta','alpha','gamma','R2'});
T_out=sortrows(T_out,'M_final','descend');
writetable(T_out,'cities_params.csv');

function c=smooth_cum(ci)
    c=double(ci(:)); jt=0; jv=0.0;
    for t=1:numel(ci), pr=0; if t>1, pr=ci(t-1); end; if ci(t)>pr, jt(end+1)=t; jv(end+1)=double(ci(t)); end; end
    for kk=1:numel(jt)-1
        t1=jt(kk); c1=jv(kk); t2=jt(kk+1); c2=jv(kk+1);
        for t=max(t1,1):t2, c(t)=c1+(c2-c1)*(t-t1)/(t2-t1); end
    end
end
function s=sse_free(le,lb_,la,tau,c_obs)
    e=exp(le); b=exp(lb_); a=exp(la);
    if e<=0||b<=0||a<=0, s=1e15; return; end
    cp=(1+b*tau.^a).^(e/b)-1; if any(~isfinite(cp)), s=1e15; return; end
    s=sum((c_obs-cp).^2);
end
function s=sse_prior(le,lb_,la,tau,c_obs,lam,sl,in,logMk,lam_beta,log_beta_lb)
    e=exp(le); b=exp(lb_); a=exp(la);
    if e<=0||b<=0||a<=0, s=1e15; return; end
    cp=(1+b*tau.^a).^(e/b)-1; if any(~isfinite(cp)), s=1e15; return; end
    s=sum((c_obs-cp).^2); if ~isfinite(s), s=1e15; return; end
    if lam>0, s=s+lam*(le-(sl*logMk+in))^2; end
    s=s+lam_beta*max(0,log_beta_lb-lb_)^2;
end
function r2=r2_score(obs,pred)
    ss_r=sum((obs-pred).^2); ss_t=sum((obs-mean(obs)).^2);
    if ss_t<1e-10, r2=0; else, r2=1-ss_r/ss_t; end
end
