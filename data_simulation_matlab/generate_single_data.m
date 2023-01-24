function [P_output, Q_output] = generate_single_data(Va_L, Va_R, Va_T, Va_E, Va_Rp, SL)
% Randomness of Modeling parameter COV (Normal Distribution)
COV_L = 1e-2; 
COV_R = 1e-2;
COV_T = 1e-2;

% Randomness of Modeling parameter COV (Log_normal Distribution)
COV_E = 1e-2; 
COV_Phai = 1e-2;
COV_Rp0 = 1e-2;
COV_Cp0 = 1e-2;
% Input Flow
handles.Param.HR = 70; % Heart Rate (bpm)
handles.Param.Mean_Flow = 70; % Mean Flow (ml/s)
handles.Param.LVET = 0.3; % LVET (s)

% Structural (Modeling) Parameter (the times of)
handles.Param.P_L = Va_L * (1 + normrnd(0, 1) * COV_L); % Body Height
handles.Param.P_R = Va_R * (1 + normrnd(0, 1) * COV_R); % Artery Radius
handles.Param.P_T = Va_T * (1 + normrnd(0, 1) * COV_T); % Wall Thickness

% Functional (Physical) parameter (the times of)
handles.Param.P_E = Va_E * lognrnd(log(1/sqrt(COV_E^2+1)), sqrt(log(COV_E^2+1))); % Young's Modulus
handles.Param.P_Phai = lognrnd(log(1/sqrt(COV_Phai^2+1)), sqrt(log(COV_Phai^2+1))); % Arterial Viscosity
handles.Param.P_Rp0 = Va_Rp * lognrnd(log(1/sqrt(COV_Rp0^2+1)), sqrt(log(COV_Rp0^2+1))); % Peripheral Resistance
handles.Param.P_Cp0 = lognrnd(log(1/sqrt(COV_Cp0^2+1)), sqrt(log(COV_Cp0^2+1))); % Peripheral Compliance

% what are these?
handles.Param.logx = 1;

%%% Run
Model = main_simulation(handles, SL);
P_output = Model.Poutput{1,1};
Q_output = Model.Qoutput{1,1};
end

function Model=main_simulation(handles, SL)
Param=handles.Param;
E=Param.P_E;
phi=Param.P_Phai;
R=Param.P_Rp0;
K=1;
NO=[1,2,10,12,13,25,27,29,31,33,49,50,53,55];
NO_arm=[1,2,10,15,17,19];
NO_carodit=[1,2,11];
F_range=2:512;
L_NO=length(NO);
HR=Param.HR;
flag_E=0;%0 represent constant E, 1 represent E changing with w0;
for i=1:length(HR) %����
    [Poutput{i},Qoutput{i},time{i},Pinput{i},Qinput{i},timeall{i},PP_mult,f_new,Pinput_fft{i},f_fft{i},Z_INPUT,P_fft{i},P{i},L_tree,node,dist]=main_etm_55_all0(handles,HR(i),E,phi,K,R,flag_E, SL);
    for k=1:L_NO
        [L(i,k),Timedelay_tan(i,k),PWV_tan(i,k)]=PWV_extraction_Tan_Min2(HR(i),K,phi,E,R,Poutput{i},Qoutput{i},time{i},P{i},NO,k,L_tree);
    end
end

%brachial
[L_b,Timedelay_tan_b,PWV_tan_b]=PWV_extraction_Tan_Min2(HR,K,phi,E(i),R,Poutput{i},Qoutput{i},time{i},P{i},NO_arm,5,L_tree);
%carodit
[L_c,Timedelay_tan_c,PWV_tan_c]=PWV_extraction_Tan_Min2(HR,K,phi,E(i),R,Poutput{i},Qoutput{i},time{i},P{i},NO_carodit,3,L_tree);

N_stat=[1,11,50,17,19,55];
SBP=max(Poutput{i}(N_stat,:),[],2);
DBP=min(Poutput{i}(N_stat,:),[],2);
PP=SBP-DBP;
MBP=mean(Poutput{i}(N_stat,11:end),2);
SBF=max(Qoutput{i}(N_stat,:),[],2);
DBF=min(Qoutput{i}(N_stat,:),[],2);
PBF=SBF-DBF;
MBF=mean(Qoutput{i}(N_stat,11:end),2);
MBF(1,1)=Param.Mean_Flow;
BP_BF=[SBP,DBP,PP,MBP,SBF,DBF,PBF,MBF];
BP_BF=BP_BF-rem(BP_BF,0.01);

cfPWV=(L(i,12)-L_c)/(Timedelay_tan(i,12)-Timedelay_tan_c);
baPWV=(L(i,14)-L_b)/(Timedelay_tan(i,14)-Timedelay_tan_b);

ABI=SBP(4)/SBP(6);

%%%%%%%%%%%%%%%%%% return result  %%%%%%%%%%%%%%%%%%%%
Model.Poutput=Poutput;
Model.Qoutput=Qoutput;
Model.time=time;
Model.Pinput=Pinput;
Model.Qinput=Qinput;
Model.timeall=timeall;
Model.Z_INPUT=Z_INPUT;
Model.PP_mult=PP_mult;
Model.f_new=f_new;
Model.Pinput_fft=Pinput_fft;
Model.f_fft=f_fft;
Model.P_fft=P_fft;
Model.P=P;
Model.L_tree=L_tree;
Model.L=L;
%Model.Timedelay=Timedelay;
Model.cfPWV=cfPWV;
Model.baPWV=baPWV;
Model.ABI=ABI;
Model.BP_BF=BP_BF;
Model.dist=dist;
Model.F_range=F_range;
end

function [Poutput,Qoutput,time,Pinput,Qinput,timeall,PP_mult,f_new,Pinput_fft,f_fft,Z_INPUT2,P_fft,P,L_tree,node1,dist]=main_etm_55_all0(handles,HR,pE,phi,kkk,pR,flag_E, SL)
Param=handles.Param;
pl=handles.Param.P_L;
pr=handles.Param.P_R;
ph=handles.Param.P_T;
pR=1.6*handles.Param.P_Rp0;
pE0=handles.Param.P_E;
pt=1;
pC=3*handles.Param.P_Cp0;
mean_flow=handles.Param.Mean_Flow;
LVET=handles.Param.LVET;
logx=handles.Param.logx;

global tree0 node Blood_D Blood_V w N_PP Z0 Z_INPUT flag_age;
%N_PP is the number and transmission ratio of pressure
Blood_D=1.05*10^3;%Blood density(kg/m^3)
Blood_V=0.0035;%Blood viscosity(pa.s)(2.28��1.11)*10^-3Pa��s
[tree0,node]=create_tree55_2009();%create the 55 branched model of arterial system; according to the paper in 2009
node(:,1)=node(:,1)/(100)*0.9*pl;%l(cm) to IU
node(:,2)=node(:,2)/(100)*1.15*pr;%r(cm) to IU
node(:,3)=node(:,3)/(100)*1.0*ph;%h(cm) to IU
flag_age=0;
if (flag_age==1)
    node(:,4)=node(:,4)*(10^6)*pE;%E to IU(N/m)
    node(:,5)=node(:,5)*(10^9)*0.5;%Rp to IU(pa.s/m)
    node(:,6)=node(:,6)*(10^-10)*0.1;%C(m^3Pa-1)
    node(:,7)=node(:,7)*(10^9)*0.17*1.2*pR;%R0
    node(:,8)=node(:,8)*(10^9)*0.17*1.2*pR;%R1
    node(:,9)=node(:,9)*(10^-10)*3*pC;%sqrt(fE);%C(m^3Pa-1)
else
    node(:,4)=node(:,4)*(10^6)*2*pE;%E to IU(N/m)2
    node(:,5)=node(:,5)*(10^9)*0.9*pR;%Rp to IU(pa.s/m) 0.9
    node(:,6)=node(:,6)*(10^-10)*1*pC;%C(m^3Pa-1) 0.1
    node(:,7)=node(:,7)*(10^9)*0.17*pR;%R0 0.17
    node(:,8)=node(:,8)*(10^9)*0.17*pR;%R1 0.17
    node(:,9)=node(:,9)*(10^-10)*3*pC;%sqrt(fE);%C(m^3Pa-1) 3
end

L_tree=node(:,1);
DD=1;
f0=32*DD;d_f0=512*DD;%f0 is the max frequency,d_f0 is the number of interval
Ntime_f=d_f0/f0;
w0=f0*[eps,1/d_f0:1/d_f0:1-1/d_f0];%the series frequencies , the frequency interval is 64/512=1/8Hz and 512 points;
f_new=w0;
w=2*pi*w0;%angle frequencies

L_w=length(w);
N_PP=zeros(55,L_w);
Z0=zeros(55,L_w);%character impedances

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  input impedance %%%%%%%%%%%%%%%%%
Z_INPUT=zeros(55,L_w);%input impedances
impedance(1,phi,kkk,flag_E,logx, SL);%computation from first node
Z_INPUT2=Z_INPUT;

%%%%%%%%%%%%%%%%  load input flow and calculate input pressure %%%%%%%%%%%%%%%%
load Q_inputwave2.mat %load flow wave
f=f_changing_HR(f,HR,LVET);
f=chazhi(f,d_f0,mean_flow);%@@@@@@@@@@@@@@@@@@@@@Interpolation and resampleing (Fs is 2*f_max Hz)

f=f*10^(-6);%change into IU
L_f=length(f);
f_fft=fft(f,Ntime_f*d_f0)./(Ntime_f*d_f0);%@@@@@@@@@@@@@@@@@@@@@@the frequency interval is d_f0/(8*d_f0)=1/8Hz ;
f2=real(ifft([f_fft(1:d_f0+1),fliplr(conj(f_fft(2:d_f0)))]*2*d_f0));%%%%%%%%%%% 512 points
P_fft=f_fft(1:d_f0).*Z_INPUT(1,:);
P_fft2=[P_fft(1:end),P_fft(end),fliplr(conj(P_fft(2:end)))];
P=real(ifft(P_fft2*2*d_f0,2*d_f0))/133.29;


%%%%%%%%%%%%%%%%%%%%%%  ���㴫��ϵ��  %%%%%%%%%%%%%%%%%%%%%%
for ii=1:55
    list_node=search_node(ii);
    PP_mult(ii,:)=ones(1,size(N_PP,2));
    dist(ii)=0;
    for iii=1:length(list_node)%the propergation ratio of all node are multplied together
        PP_mult(ii,:)=PP_mult(ii,:).*N_PP(list_node(iii),:);%%%%%%55*100��������
        dist(ii)=dist(ii)+node(list_node(iii),1);
    end
end
P_fft=ones(55,1)*P_fft;
Pinput_fft=PP_mult.*P_fft;
Pinput=real(ifft([Pinput_fft,0*Pinput_fft(:,1),fliplr(conj(Pinput_fft(:,2:end)))].*2*d_f0,[],2));
Qinput_fft=Pinput_fft./Z_INPUT;
Qinput=real(ifft([Qinput_fft,0*Qinput_fft(:,1),fliplr(conj(Qinput_fft(:,2:end)))].*2*d_f0,[],2));
Pinput=Pinput/133.29;%change into ml
Qinput=Qinput*10^6;%change into ml
f=f*10^(6);

interval=round((60/HR)/(1/(2*f0)));%%%%%%%%%%%%%%%%
startpoint=5*interval-10;
% interval=round(d_f0/3*60/HR);
Low1=searchstartpoint(startpoint,Pinput(1,startpoint:startpoint+interval));

N_T=Low1-10:Low1+interval;
Poutput=Pinput(:,N_T);
Qoutput=Qinput(:,N_T);
P=P(N_T);
time=N_T/(2*f0);
timeall=(1:size(Pinput,2))/(2*f0);


node1=node;

end

function [L,timedelay,PWV]=PWV_extraction_Tan_Min2(N,K,fai,E,R,Poutput,Qoutput,time,P,NO,k,L_tree)
NO_all=NO;
NO=NO(k);
%The extaction method of the Point based on the mean of aorta pressure
P1=P(1,:);
P50=Poutput(NO,:);
Q1=Qoutput(1,:);
Q50=Qoutput(NO,:);
plot(P50)
%%%%%%%%%%%%%interpolation%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%Finding the minimum point%%%%%%%%%%%%%
T=time(1)+60/N;
[V,ind10]=min(P1(T>time));
Point1=P1(ind10);
[V,ind20]=min(P50(T>time));
Point2=P50(ind20);

%%%%%%%%%%%%%%Finding the foot point of P1 %%%%%%%%%%%%%%
[max_P1,ind_max_P1]=max(P1(T>time));
sub=P1(2:ind_max_P1)-P1(1:ind_max_P1-1);
[v,ind1]=max(sub);%find out the sheepest point index
T1=time(ind1+1)-(P1(ind1+1)-Point1)*((time(ind1+1)-time(ind1))/(P1(ind1+1)-P1(ind1)));

%%%%%%%%%%%%%%Finding the foot point of P50 %%%%%%%%%%%%%%
[max_P50,ind_max_P50]=max(P50(T>time));
sub=P50(2:ind_max_P50)-P50(1:ind_max_P50-1);
[v,ind2]=max(sub);%find out the sheepest point index
T2=time(ind2+1)-(P50(ind2+1)-Point2)*((time(ind2+1)-time(ind2))/(P50(ind2+1)-P50(ind2)));

%%%%%%%%%%%%%%Calculate the PWV%%%%%%%%%%%%%%%%%%%
L=sum(L_tree(NO_all(NO_all<=NO)));%L is the distance from aorta to femoral
PWV=L/(T2-T1);
timedelay=T2-T1;
%%%%%%%%%%%%%%Plot the results%%%%%%%%%%%%%%%%%%%%%%
% plot(time,P1,'.-',time,P50,'.-');
% hold on
% plot([T1 time(ind10)],[Point1 Point1]);%plot horizontal line
% plot([T2 time(ind20)],[Point2 Point2]);%plot horizontal line
% plot([T1 time(ind1+1)],[Point1 P1(ind1+1)]);%plot slop line
% plot([T2 time(ind2+1)],[Point2 P50(ind2+1)]);%plot slop line
% plot(T1,Point1,'ro',T2,Point2,'ro');


%%%%%%%%%%%%%%%%%%Calculating and Showing some information%%%%%%%%%%%%%%
ind_temp=find(T>time);
ind_period=ind_temp(end);
mean_P1=mean(P1(1:ind_period));
mean_Q1=mean(Q1(1:ind_period));

%disp(['NO = ' num2str(NO) ',HR = ' num2str(N) ', PWV = ' num2str(PWV) ', K = ' num2str(K) ', E = ' num2str(E) ',  phi = ' num2str(fai)  ', Mean of Flow: ' num2str(mean_Q1),', Mean of Pressure: ' num2str(mean_P1)]);
end

function [L,timedelay,PWV]=PWV_extraction_Max_first_derivative2(N,K,fai,E,R,Poutput,Qoutput,time,P,NO,k,L_tree)
NO_all=NO;
NO=NO(k);
%The extaction method of the Point based on the sharpest slope
P1_0=P(1,:);
P50_0=Poutput(NO,:);
Q1_0=Qoutput(1,:);
Q50_0=Qoutput(NO,:);
%%%%%%%%%%%%%  interpolation  %%%%%%%%%%%%%%%%%%%
time0=time;
time=(min(time0):1/400:max(time0));
P1=(interp1(time0,P1_0,time,'spline'));
P50=(interp1(time0,P50_0,time,'spline'));
Q1=(interp1(time0,Q1_0,time,'spline'));
Q50=(interp1(time0,Q50_0,time,'spline'));


%%%%%%%%%%%%%%  Finding the foot point of P1   %%%%%%%%%%%%%%
T=time(1)+60/N;
[max_P1,ind_max_P1]=max(P1(T>time));
sub=P1(2:ind_max_P1)-P1(1:ind_max_P1-1);
[v,ind1]=max(sub);%find out the sheepest point index
T1=time(ind1);
Point1=P1(ind1);

%%%%%%%%%%%%%%  Finding the foot point of P50  %%%%%%%%%%%%%%
[max_P50,ind_max_P50]=max(P50(T>time));
sub=P50(2:ind_max_P50)-P50(1:ind_max_P50-1);
[v,ind2]=max(sub);%find out the sheepest point index
T2=time(ind2);
Point2=P50(ind2);

%%%%%%%%%%%%%%  Calculate the PWV  %%%%%%%%%%%%%%%%%%%
L=sum(L_tree(NO_all(NO_all<=NO)));%L is the distance from aorta to a certain artery
PWV=L/(T2-T1);
timedelay=T2-T1;
%%%%%%%%%%%%%%Plot the results%%%%%%%%%%%%%%%%%%%%%%
if(0)
    plot(time,P1,'.-',time,P50,'.-');
    hold on
    plot(T1,Point1,'ro',T2,Point2,'ro');
end

%%%%%%%%%%%%%%%%%%Calculating and Showing some information%%%%%%%%%%%%%%
ind_temp=find(T>time);
ind_period=ind_temp(end);
mean_P1=mean(P1(1:ind_period));
mean_Q1=mean(Q1(1:ind_period));
%disp(['HR = ' num2str(N) ', PWV = ' num2str(PWV) ', K = ' num2str(K) ', E = ' num2str(E) ',  phi = ' num2str(fai)  ', Mean of Flow: ' num2str(mean_Q1),', Mean of Pressure: ' num2str(mean_P1)]);
end

function z=impedance(n,phi,kkk,flag_E,logx, SL)
global tree0 node Blood_D Blood_V w N_PP Z0 Z_INPUT flag_age DL;
%calculate the characteristic impedance and propagation constant
l=node(n,1);r=node(n,2);h=node(n,3);E=node(n,4);
E=Set_E1(r,E,0);%adjustment of E
E=Set_E2(r,E,0);%adjustment of E

%%Stsnosis
if n==33 %[1,2,10,13,25,31,49,50,53,55]
    r0=r;
    r=sqrt(SL*(r^2));%stenosis fraction
    h=h+(r0-r);
end
%[Z0(n,:),gama]=Z0_fun1(l,r,h,E,Blood_D,Blood_V,w);%characteristic impedance
%[Z0(n,:),gama]=Z0_fun2(l,r,h,E,Blood_D,Blood_V,w);
%[Z0(n,:),gama]=Z0_fun2_1(l,r,h,E,Blood_D,Blood_V,w);

if (flag_E==0)
    [Z0(n,:),gama]=Z0_fun3(l,r,h,E,Blood_D,Blood_V,w,phi,kkk);
else
    [Z0(n,:),gama]=Z0_fun4(l,r,h,E,Blood_D,Blood_V,w,phi,kkk,logx);%E changing with frequency
end
%calculate the impedance Zt and reflection coefficient
segment=tree0{n};
k=size(segment,2);
%Initialization
if k==1&&segment(k)==0
    Ref='crr';
    switch lower(Ref)
        case 'rp'%ֻ���ǵ���
            ZL=node(n,5);
            Reflection_C=(ZL-Z0(n,:))./(ZL+Z0(n,:));
            z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
            N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
            Z_INPUT(n,:)=z;
        case 'cr'%����ĩ�˵���
            ZL=1./(1./node(n,5)+j*w*node(n,6));%����
            Reflection_C=(ZL-Z0(n,:))./(ZL+Z0(n,:));
            z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
            N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
            Z_INPUT(n,:)=z;
        case 'crr'%����ĩ�˵��ݺ�˫����
            ZL=node(n,7)+1./(1./node(n,8)+j*w*node(n,9)*10);%����R0,����R1��C
            Reflection_C=(ZL-Z0(n,:))./(ZL+Z0(n,:));
            z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
            N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
            Z_INPUT(n,:)=z;
        case 'c0'%���÷���ϵ��
            Reflection_C=0.8;
            z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
            N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
            Z_INPUT(n,:)=z;
        case 'c1'%���÷���ϵ��
            ZL=8*Blood_V*l/(pi*r^4)*200;
            Reflection_C=(ZL-Z0(n,:))./(ZL+Z0(n,:));
            z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
            N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
            Z_INPUT(n,:)=z;
    end
else
    for m=1:k
        ZL0(m,:)=impedance(segment(m),phi,kkk,flag_E,logx, SL);
    end
    ZL=1./sum(1./ZL0,1);
    Reflection_C=(ZL-Z0(n,:))./(ZL+Z0(n,:));
    z=Z0(n,:).*(1+Reflection_C.*exp(-2*l*gama))./(1-Reflection_C.*exp(-2*l*gama));
    %z=Z0(n,:).*((ZL+Z0(n,:).*tanh(gama*l))./(Z0(n,:)+ZL.*tanh(gama*l)));
    N_PP(n,:)=(1+Reflection_C)./(exp(gama*l)+Reflection_C.*exp(-gama*l));
    Z_INPUT(n,:)=z;
end
end

function E=Set_E1(r,E,flag)
if flag==1
    kE=4;
    if r<=0.0025
        E=kE*4e5;
    elseif r>0.0025 && r<=0.004
        E=kE*3e5;
    else
        E=kE*2e5;
    end
end
end

function E=Set_E2(r,E,flag)
if flag==1
    if E==400000
        kE=1.5;
    elseif E==800000
        kE=1.4;
    elseif E==1600000
        kE=1.2;
    end
    E=kE*E;
end
end

function [ZZ0,gama]=Z0_fun1(l,r,h,E,B_D,B_V,w)
C=(3*pi*r^3)/(2*E*h);
G=0;
if r>0.002
    lambdaQ=0.2057;
    lambdaP=0.0392;
    R=(8*B_V)/(pi*r^4);%*l
    L=(lambdaQ-lambdaP)*(8*B_D)/(pi*r^2);
    ZZ0=sqrt((R+j*w*L)./(G+j*w*C));
    gama=sqrt((R+j*w*L).*(G+j*w*C));
else
    lambdaQ=0.1729;
    lambdaP=0.0075;
    R1=(lambdaQ/lambdaP-1)*(8*B_V)/(pi*r^4);
    R2=(8*B_V)/(pi*r^4);%*l
    L=(lambdaQ-lambdaP)*(8*B_D)/(pi*r^2);
    R1L=1./(1/R1+1./(j*w*L));
    ZZ0=sqrt((R2+R1L)./(G+j*w*C));
    gama=sqrt((R2+R1L).*(G+j*w*C));
end
end

function [ZZ0,gama]=Z0_fun2(l,r,h,E,B_D,B_V,w)
C=(3*pi*r^3)/(2*E*h);
G=0;
m=5;
k=0.1*(1.2-1);
R0=(8*B_V)/(pi*r^4)/(4*k);
L=(B_D./(pi*r^2)./(2*[1:m]-1))'*j*w;
R=(8*B_V)/(pi*r^4)*[1:m];%*l
for i=m:-1:3
    L(i-1,:)=1./(1./(L(i,:)+R(i-1))+1./(L(i-1,:)));
end
%RL=R(1)+L(2,:);
RL=1./(1/R0+1./(R(1)+L(2,:)));%Anomalous viscosity of Blood
ZZ0=sqrt((L(1,:)+RL)./(G+j*w*C));
gama=sqrt((L(1,:)+RL).*(G+j*w*C));
end

function [ZZ0,gama]=Z0_fun2_1(l,r,h,E,B_D,B_V,w)
C=(3*pi*r^3)/(2*E*h);
G=0;
rafa=r*sqrt(w*B_D/B_V);
lamda=(j*r^2*w*B_D)/(4*B_V);
n=10;
ab=0;
for m=n:-1:3
    if mod(m,2)~=0
        ab=1./(m./lamda+ab);
    else
        ab=1./(m+ab);
    end
end
ab=4*B_V/(pi*r^4)*(lamda+2+ab);
ZZ0=sqrt(ab./(G+1j*w*C));
gama=sqrt(ab.*(G+1j*w*C));
end

function [ZZ0,gama]=Z0_fun3(l,r,h,E,B_D,B_V,w,phi0,kkk)
c0=sqrt(E*h/(B_D*2*r));
Poisson_R=0.5;
alpha=r*sqrt(w*B_D/B_V);
phai=phi0*(10*h/(2*r))*(pi/180).*((1-exp(-kkk*w))/2);
%phai=2*phi0*(5*h/r)*(pi/180).*((1-exp(-kkk*w))/2);
%phai=10*(pi/180)*((1-exp(-2*w))/2);

F10=(2*besselj(1,alpha*1i^(1.5)))./(alpha*1i^(1.5).*besselj(0,alpha*1i^(1.5)));
F10(1)=F10(1)+eps;
ZZ0=(B_D*c0/(sqrt(1-Poisson_R^2))).*((1-F10).^(-0.5)).*(cos(phai)+1i*sin(phai))/(pi*r^2);
%ZZ0=(B_D*c0/(pi*r^2*(sqrt(1-Poisson_R^2)))).*((1-F10).^(0.5)).*(cos(phai)+i*sin(phai));
%ZZ0=B_D*c0/(pi*r^2);
%ZZ0=B_D*c0/(pi*r^2)/(sqrt(1-Poisson_R^2));
%ZZ0=B_D*c0/(pi*r^2)/(sqrt(1-Poisson_R^2)).*((1-F10).^(-0.5));
%gama=j*w/c0.*(1-F10).^(-0.5).*(cos(phai)-i*sin(phai));
gama=1i*w*((1-Poisson_R^2).^(1/2))/c0.*((1-F10).^(-0.5)).*(cos(phai)-1i*sin(phai));
end

function [ZZ0,gama]=Z0_fun4(l,r,h,E0,B_D,B_V,w,phi0,kkk,logx)
%set a E ralative to w
if (logx==0)
    log_E=log([w(2:end),w(end)]);
    %log_E=1;
    log_mu=log([w(2:end),w(end)]);
    % log_E=fliplr(log([w(2:end),w(end)]));
    %log_mu=fliplr(log([w(2:end),w(end)]));
    mu0=10000;
    E=(E0*0.9+E0/10*log_E);%+(1j*(mu0+mu0/2*log_mu));
    phi0=phi0*exp([w(2:end),w(end)]/w(end));
else
    [E,phi0]=constru(w,E0,phi0,logx);
end


c0=sqrt(E*h/(B_D*2*r));
Poisson_R=0.5;
alpha=r*sqrt(w*B_D/B_V);
phai=phi0.*(5*h/r)*(pi/180).*((1-exp(-kkk*w))/2);
%phai=10*(pi/180)*((1-exp(-2*w))/2);

F10=(2*besselj(1,alpha*1i^(1.5)))./(alpha*1i^(1.5).*besselj(0,alpha*1i^(1.5)));
F10(1)=F10(1)+eps;
ZZ0=(B_D*c0./(sqrt(1-Poisson_R^2))).*((1-F10).^(-0.5)).*(cos(phai)+1i*sin(phai))/(pi*r^2);
%ZZ0=(B_D*c0/(pi*r^2*(sqrt(1-Poisson_R^2)))).*((1-F10).^(0.5)).*(cos(phai)+i*sin(phai));
%ZZ0=B_D*c0/(pi*r^2);
%ZZ0=B_D*c0/(pi*r^2)/(sqrt(1-Poisson_R^2));
%ZZ0=B_D*c0/(pi*r^2)/(sqrt(1-Poisson_R^2)).*((1-F10).^(-0.5));
%gama=j*w/c0.*(1-F10).^(-0.5).*(cos(phai)-i*sin(phai));
gama=1i*w*((1-Poisson_R^2).^(1/2))./c0.*((1-F10).^(-0.5)).*(cos(phai)-1i*sin(phai));
end

function [E,phi]=constru(w,E0,phi0,logx)
%construct the E depended on frequence
N_w=length(w);
N_2Hz=length(find(w/(2*pi)<=30));%4 Hz
temp=[2,5,10,100,1000];
y=log(linspace(1,temp(logx),N_2Hz));
y=y/max(y);
E0=repmat(E0,1,N_w);
E=E0;
E(1:N_2Hz)=y.*E0(1:N_2Hz);
E(1)=eps;
E=E*4/4+E(end)*2/4;
phi=phi0;%*E/E(end);
%phi=phi0*E/E(end);
end

function list_node=search_node(aim)
%search the path to the end node
global tree0
list_node=aim;
n=size(tree0,2);
for mm=1:n
    ii=n-mm+1;
    if(find(tree0{ii}==aim))
        list_node=[ii,list_node];
        aim=ii;
    end
end
end

function yi=chazhi(f,N,mean_flow)
x=f(:,1);
y=f(:,2);
xi=(min(x):1/N:max(x));
yi=(interp1(x,y,xi,'spline'));
yi=mean_flow/mean(yi)*yi; %control the same flow

yi=repmat(yi,1,20);% period extension
% xi2=(min(x):1/N:length(yi)/N);
% yi2=(interp1(xi2,yi,xi2+1/(2*N),'cubic'));
%
% figure;plot(yi2);title('Input Flow Wave');xlabel('time(s)');ylabel('flow(ml/s)');
end

function p=normalization1(p)
p_max=max(p,[],2);
p_min=min(p,[],2);
ones1=ones(1,size(p,2));
p=(2*p-(p_max+p_min)*ones1)./((p_max-p_min)*ones1);
end

function startpoint=searchstartpoint(N,wave)
[a,startpoint]=min(wave);
startpoint=startpoint+N;
end

function [tree,node]=create_tree55_2009
tree{1}=[2,3];
tree{2}=[10,11];
tree{3}=[4,5];
tree{4}=[6,7];
tree{5}=[39,47];
tree{6}=0;
tree{7}=[8,9];
tree{8}=0;
tree{9}=[43,44];
tree{10}=[12,15];
tree{11}=[40,48];
tree{12}=[13,14];
tree{13}=[20,25];
tree{14}=0;
tree{15}=[16,17];
tree{16}=0;
tree{17}=[18,19];
tree{18}=[45,46];
tree{19}=0;
tree{20}=[21,22];
tree{21}=[24,23];
tree{22}=0;
tree{23}=0;
tree{24}=0;
tree{25}=[26,27];
tree{26}=0;
tree{27}=[29,30];
tree{28}=0;
tree{29}=[28,31];
tree{30}=0;
tree{31}=[32,33];
tree{32}=0;
tree{33}=[34,49];
tree{34}=[35,36];
tree{35}=[37,38];
tree{36}=0;
tree{37}=0;
tree{38}=[41,42];
tree{39}=0;
tree{40}=0;
tree{41}=0;
tree{42}=0;
tree{43}=0;
tree{44}=0;
tree{45}=0;
tree{46}=0;
tree{47}=0;
tree{48}=0;
tree{49}=[50,51];
tree{50}=[52,53];
tree{51}=0;
tree{52}=0;
tree{53}=[54,55];
tree{54}=0;
tree{55}=0;
%������
node=[2,1.47500000000000,0.163000000000000,0.400000000000000,0,0,0,0,0;3,1.38000000000000,0.126000000000000,0.400000000000000,0,0,0,0,0;3.50000000000000,0.635000000000000,0.0800000000000000,0.400000000000000,0,0,0,0,0;3.50000000000000,0.420000000000000,0.0670000000000000,0.400000000000000,0,0,0,0,0;17.7000000000000,0.385000000000000,0.0630000000000000,0.400000000000000,0,0,0,0,0;13.5000000000000,0.200000000000000,0.0450000000000000,0.800000000000000,6.01000000000000,0.620000000000000,6.10000000000000,27.8700000000000,0.0126000000000000;39.8000000000000,0.320000000000000,0.0670000000000000,0.400000000000000,0,0,0,0,0;22,0.160000000000000,0.0430000000000000,0.800000000000000,5.28000000000000,0.700000000000000,14.2100000000000,18.3400000000000,0.0143000000000000;6.70000000000000,0.220000000000000,0.0460000000000000,0.800000000000000,0,0,0,0,0;4,1.29500000000000,0.115000000000000,0.400000000000000,0,0,0,0,0;20.8000000000000,0.385000000000000,0.0630000000000000,0.400000000000000,0,0,0,0,0;5.50000000000000,1.18500000000000,0.110000000000000,0.400000000000000,0,0,0,0,0;10.5000000000000,1.02000000000000,0.110000000000000,0.400000000000000,0,0,0,0,0;7.30000000000000,0.300000000000000,0.0490000000000000,0.400000000000000,1.39000000000000,2.68000000000000,2,6.04000000000000,0.0542000000000000;3.50000000000000,0.420000000000000,0.0660000000000000,0.400000000000000,0,0,0,0,0;13.5000000000000,0.200000000000000,0.0450000000000000,0.800000000000000,6.01000000000000,0.620000000000000,6.10000000000000,27.8700000000000,0.0126000000000000;39.8000000000000,0.320000000000000,0.0670000000000000,0.400000000000000,0,0,0,0,0;6.70000000000000,0.220000000000000,0.0460000000000000,0.800000000000000,0,0,0,0,0;22,0.160000000000000,0.0430000000000000,0.800000000000000,5.28000000000000,0.700000000000000,14.2100000000000,18.3400000000000,0.0143000000000000;2,0.325000000000000,0.0640000000000000,0.400000000000000,0,0,0,0,0;2,0.275000000000000,0.0640000000000000,0.400000000000000,0,0,0,0,0;6.50000000000000,0.265000000000000,0.0490000000000000,0.400000000000000,3.63000000000000,1.02000000000000,2.80000000000000,17.4800000000000,0.0208000000000000;5.80000000000000,0.165000000000000,0.0540000000000000,0.400000000000000,5.41000000000000,0.690000000000000,8.59000000000000,22.9700000000000,0.0139000000000000;5.50000000000000,0.200000000000000,0.0450000000000000,0.400000000000000,2.32000000000000,1.60000000000000,4.05000000000000,9.59000000000000,0.0325000000000000;5.30000000000000,0.880000000000000,0.0900000000000000,0.400000000000000,0,0,0,0,0;5,0.375000000000000,0.0690000000000000,0.400000000000000,0.930000000000000,4,1.20000000000000,4.15000000000000,0.0810000000000000;1.50000000000000,0.825000000000000,0.0800000000000000,0.400000000000000,0,0,0,0,0;3,0.280000000000000,0.0530000000000000,0.400000000000000,1.13000000000000,3.29000000000000,2.02000000000000,4.64000000000000,0.0667000000000000;1.50000000000000,0.800000000000000,0.0800000000000000,0.400000000000000,0,0,0,0,0;3,0.280000000000000,0.0530000000000000,0.400000000000000,1.13000000000000,3.29000000000000,2.02000000000000,4.64000000000000,0.0667000000000000;12.5000000000000,0.710000000000000,0.0750000000000000,0.400000000000000,0,0,0,0,0;3.80000000000000,0.190000000000000,0.0430000000000000,0.400000000000000,6.88000000000000,0.540000000000000,5.37000000000000,33.1000000000000,0.0110000000000000;8,0.590000000000000,0.0650000000000000,0.400000000000000,0,0,0,0,0;5.80000000000000,0.385000000000000,0.0600000000000000,0.400000000000000,0,0,0,0,0;14.5000000000000,0.340000000000000,0.0530000000000000,0.800000000000000,0,0,0,0,0;4.50000000000000,0.200000000000000,0.0400000000000000,1.60000000000000,7.94000000000000,0.470000000000000,6.40000000000000,25.2800000000000,0.0136000000000000;11.3000000000000,0.200000000000000,0.0470000000000000,0.800000000000000,4.77000000000000,0.780000000000000,4.99000000000000,14.4000000000000,0.0226000000000000;44.3000000000000,0.295000000000000,0.0500000000000000,0.800000000000000,0,0,0,0,0;17.7000000000000,0.200000000000000,0.0420000000000000,0.800000000000000,13.9000000000000,0.270000000000000,5.23000000000000,23.5300000000000,0.0148000000000000;17.6000000000000,0.290000000000000,0.0450000000000000,0.800000000000000,13.9000000000000,0.270000000000000,2.53000000000000,25.4200000000000,0.0148000000000000;34.4000000000000,0.180000000000000,0.0450000000000000,1.60000000000000,4.77000000000000,0.780000000000000,9.90000000000000,32.7800000000000,0.0102000000000000;32.2000000000000,0.250000000000000,0.0390000000000000,1.60000000000000,5.59000000000000,0.670000000000000,3.96000000000000,15.1100000000000,0.0226000000000000;7,0.100000000000000,0.0280000000000000,1.60000000000000,84.3000000000000,0.0400000000000000,39.4300000000000,424.010000000000,0.000900000000000000;17,0.190000000000000,0.0460000000000000,0.800000000000000,5.28000000000000,0.700000000000000,10.1200000000000,21.2000000000000,0.0143000000000000;17,0.190000000000000,0.0460000000000000,0.800000000000000,5.28000000000000,0.700000000000000,10.1200000000000,21.2000000000000,0.0143000000000000;7,0.100000000000000,0.0280000000000000,1.60000000000000,84.3000000000000,0.0400000000000000,39.4300000000000,424.010000000000,0.000900000000000000;17.6000000000000,0.290000000000000,0.0450000000000000,0.800000000000000,13.9000000000000,0.270000000000000,2.53000000000000,25.4200000000000,0.0148000000000000;17.7000000000000,0.200000000000000,0.0420000000000000,0.800000000000000,13.9000000000000,0.270000000000000,5.23000000000000,23.5300000000000,0.0148000000000000;5.80000000000000,0.385000000000000,0.0600000000000000,0.400000000000000,0,0,0,0,0;14.5000000000000,0.340000000000000,0.0530000000000000,0.800000000000000,0,0,0,0,0;4.50000000000000,0.200000000000000,0.0400000000000000,1.60000000000000,7.94000000000000,0.470000000000000,6.40000000000000,25.2800000000000,0.0136000000000000;11.3000000000000,0.200000000000000,0.0470000000000000,0.800000000000000,4.77000000000000,0.780000000000000,4.99000000000000,14.4000000000000,0.0226000000000000;44.3000000000000,0.295000000000000,0.0500000000000000,0.800000000000000,0,0,0,0,0;32.2000000000000,0.180000000000000,0.0450000000000000,1.60000000000000,4.77000000000000,0.780000000000000,9.90000000000000,32.7800000000000,0.0102000000000000;34.4000000000000,0.250000000000000,0.0390000000000000,1.60000000000000,5.59000000000000,0.670000000000000,3.96000000000000,15.1100000000000,0.0226000000000000;];
end

function f=f_changing_HR(f,HR,LVET)
if LVET==0
    LVET=-0.0017*HR + 0.413;
end
% LVET=-0.0017*HR + 0.413;
T0=f(55,1);%beginning of the Distol
f(1:55,1)=LVET/T0*f(1:55,1);
f(56:end,1)=(60/HR-LVET)/ (f(end,1)-T0)*(f(56:end,1)-T0)+LVET;
end
