%% main function
function [data, label] = generate_massiave_data(Inter_individuality, severity_level, sample_num)
data = zeros(length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality), length(severity_level) * sample_num, 55, 66);
label = zeros(length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality) * length(Inter_individuality), length(severity_level) * sample_num, 6);

Combinations = zeros(5, length(Inter_individuality)^5);
n = 1;
for ll = 1 : length(Inter_individuality) %Length
    for rr = 1 : length(Inter_individuality) %Diameter
        for hh = 1 : length(Inter_individuality) %thickness
            for ee = 1 : length(Inter_individuality) %youngs modulus
                for rp = 1 : length(Inter_individuality) %resistance
                    Combinations(1, n) = ll;
                    Combinations(2, n) = rr;
                    Combinations(3, n) = hh;
                    Combinations(4, n) = ee;
                    Combinations(5, n) = rp;
                    n = n + 1;
                end
            end
        end
    end
end

%% Setting (inner for loop)
parfor ii = 1: size(Combinations, 2)
    ll = Combinations(1, ii);
    rr = Combinations(2, ii);
    hh = Combinations(3, ii);
    ee = Combinations(4, ii);
    rp = Combinations(5, ii);
    tdata = zeros(length(severity_level) * sample_num, 55, 66);
    tlabel = zeros(length(severity_level) * sample_num, 6);

    for ss = 1 : length(severity_level)
        for nn = 1 : sample_num
            % parameter variation
            Va_L = Inter_individuality(ll);
            Va_R = Inter_individuality(rr);
            Va_T = Inter_individuality(hh);
            Va_E = Inter_individuality(ee);
            Va_Rp = Inter_individuality(rp);
            SL = severity_level(ss);
            
            [tdata(sample_num * (ss - 1) + nn, :, :), dummy] = generate_single_data(Va_L, Va_R, Va_T, Va_E, Va_Rp, SL);
            tlabel(sample_num * (ss - 1) + nn,:) = [ll, rr, hh, ee, rp, ss-1];
        end
    end
    data(ii, :, :, :) = tdata;
    label(ii, :,:) = tlabel;
    toc
end
data = reshape(data, [length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(severity_level) * sample_num, 55, 66]);
data = permute(data, [5, 4, 3, 2, 1, 6, 7, 8]);
label = reshape(label, [length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(Inter_individuality), length(severity_level) * sample_num, 6]);
label = permute(label, [5, 4, 3, 2, 1, 6, 7]);
end
