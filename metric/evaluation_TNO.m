clc
clear all
%names = {'new100','new200','new300','new400','new500','new600','new700','new800','new900','new1000','new1100','new1200','new1300','new1400','new1500','new1600','new1700','new1800','new1900','new2000','new2100','new2200','new2300'};
%rows = ['A', 'B','C','D','E', 'F','G','H','I','J', 'K','L','M','N', 'O','P','Q', 'R','S','T','U', 'V','W'] ;

names = {'CDDFuse', 'DIDFuse', 'MetaFusion', 'SwinFusion', 'U2Fusion', 'YUVFusion'};
rows = ['A', 'B', 'C', 'D', 'E', 'F'];
easy = 1; %% easy=1 
dataset = 'TNO'
row_name1 = 'row1';
row_data1 = 'row2';
for i = 1 : length(names)
    method_name = cellstr(names(i));
    class(names(i))
    row = rows(i);
    row_name = strrep(row_name1, 'row', row);
    row_data = strrep(row_data1, 'row', row);
    fileFolder=fullfile('/home/suguilin/baseline/dataset', dataset, 'ir'); 
    dirOutput=dir(fullfile(fileFolder,'*.*'));
    fileNames = {dirOutput.name};
    [m, num] = size(fileNames);   
    ir_dir = fullfile('/home/suguilin/baseline/dataset', dataset, 'ir'); 
    vi_dir = fullfile('/home/suguilin/baseline/dataset', dataset, 'vi'); 
    % Fused_dir = '/home/suguilin/baseline/Results';
    Fused_dir = fullfile('/home/suguilin/baseline/Results', string(names(i)), dataset); 
    EN_set = [];    SF_set = [];SD_set = [];PSNR_set = [];
    MSE_set = [];MI_set = [];VIF_set = []; AG_set = [];
    CC_set = [];SCD_set = []; Qabf_set = [];
    SSIM_set = []; MS_SSIM_set = [];
    Nabf_set = []; FMI_pixel_set = [];
    FMI_dct_set = []; FMI_w_set = [];
    for j = 1:num
        if (isequal(fileNames{j}, '.') || isequal(fileNames{j}, '..'))
            continue;
        else
            fileName_source_ir = fullfile(ir_dir, fileNames{j});
            fileName_source_vi = fullfile(vi_dir, fileNames{j}); 
            fileName_Fusion = fullfile(Fused_dir, fileNames{j});
            disp(fileName_source_ir)
            ir_image = imread(fileName_source_ir);
            vi_image = imread(fileName_source_vi);
            disp(fileName_Fusion)
            fused_image = imread(fileName_Fusion);
            if size(ir_image, 3)>2
                ir_image = rgb2gray(ir_image);
            end

            if size(vi_image, 3)>2
                vi_image = rgb2gray(vi_image);
            end

            if size(fused_image, 3)>2
                fused_image = rgb2gray(fused_image);
            end

            [m, n] = size(fused_image);
        %     fused_image = fused_image(7:m-6, 7:n-6);
            ir_size = size(ir_image);
            vi_size = size(vi_image);
            fusion_size = size(fused_image);
            disp(ir_size)
            disp(vi_size)
            disp(fusion_size)
            if length(ir_size) < 3 && length(vi_size) < 3
                [EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w] = analysis_Reference(fused_image,ir_image,vi_image, easy);
                EN_set = [EN_set, EN];SF_set = [SF_set,SF];SD_set = [SD_set, SD];PSNR_set = [PSNR_set, PSNR];
                MSE_set = [MSE_set, MSE];MI_set = [MI_set, MI]; VIF_set = [VIF_set, VIF];
                AG_set = [AG_set, AG]; CC_set = [CC_set, CC];SCD_set = [SCD_set, SCD];
                Qabf_set = [Qabf_set, Qabf]; Nabf_set = [Nabf_set, Nabf];
                SSIM_set = [SSIM_set, SSIM]; MS_SSIM_set = [MS_SSIM_set, MS_SSIM];
                FMI_pixel_set = [FMI_pixel_set, FMI_pixel]; FMI_dct_set = [FMI_dct_set,FMI_dct];
                FMI_w_set = [FMI_w_set, FMI_w];
            else
                disp('unsucessful!')
                disp( fileName_Fusion)
            end
            
            fprintf('Fusion Method:%s, Image Name: %s\n', cell2mat(names(i)), fileNames{j})
        end
    end
    save_dir = '/home/suguilin/baseline/Results/MetaFusion'; 
    if exist(save_dir,'dir')==0
        mkdir(save_dir);
    end

    file_name = fullfile(save_dir, strcat('Metric_', dataset, '.xlsx')); 
    if easy ==1
        SD_table = table(SD_set');
        PSNR_table = table(PSNR_set');
        MSE_table = table(MSE_set');
        MI_table = table(MI_set');
        VIF_table = table(VIF_set');
        AG_table = table(AG_set');
        CC_table = table(CC_set');
        SCD_table = table(SCD_set');
        EN_table = table(EN_set');
        Qabf_table = table(Qabf_set');
        SF_table = table(SF_set');
        method_table = table(method_name);
        
        writetable(SD_table,file_name,'Sheet','SD','Range',row_data);
        writetable(PSNR_table,file_name,'Sheet','PSNR','Range',row_data);
        writetable(MSE_table,file_name,'Sheet','MSE','Range',row_data);
        writetable(MI_table,file_name,'Sheet','MI','Range',row_data);
        writetable(VIF_table,file_name,'Sheet','VIF','Range',row_data);
        writetable(AG_table,file_name,'Sheet','AG','Range',row_data);
        writetable(CC_table,file_name,'Sheet','CC','Range',row_data);
        writetable(SCD_table,file_name,'Sheet','SCD','Range',row_data);
        writetable(EN_table,file_name,'Sheet','EN','Range',row_data);
        writetable(Qabf_table,file_name,'Sheet','Qabf','Range',row_data);
        writetable(SF_table,file_name,'Sheet','SF','Range',row_data);
        
        writetable(method_table,file_name,'Sheet','SD','Range',row_name);
        writetable(method_table,file_name,'Sheet','PSNR','Range',row_name);
        writetable(method_table,file_name,'Sheet','MSE','Range',row_name);
        writetable(method_table,file_name,'Sheet','MI','Range',row_name);
        writetable(method_table,file_name,'Sheet','VIF','Range',row_name);
        writetable(method_table,file_name,'Sheet','AG','Range',row_name);
        writetable(method_table,file_name,'Sheet','CC','Range',row_name);
        writetable(method_table,file_name,'Sheet','SCD','Range',row_name);
        writetable(method_table,file_name,'Sheet','EN','Range',row_name);
        writetable(method_table,file_name,'Sheet','Qabf','Range',row_name);
        writetable(method_table,file_name,'Sheet','SF','Range',row_name);
    else    
        Nabf_table = table(Nabf_set');
        SSIM_table = table(SSIM_set');
        MS_SSIM_table = table(MS_SSIM_set');
        FMI_pixel_table = table(FMI_pixel_set');
        FMI_dct_table = table(FMI_dct_set');
        FMI_w_table = table(FMI_w_set');
        method_table = table(method_name);
        
        writetable(Nabf_table,file_name,'Sheet','Nabf','Range',row_data);
        writetable(SSIM_table,file_name,'Sheet','SSIM','Range',row_data);
        writetable(MS_SSIM_table,file_name,'Sheet','MS_SSIM','Range',row_data);
        writetable(FMI_pixel_table,file_name,'Sheet','FMI_pixel','Range',row_data);
        writetable(FMI_dct_table,file_name,'Sheet','FMI_dct','Range',row_data);
        writetable(FMI_w_table,file_name,'Sheet','FMI_w','Range',row_data);
        
        writetable(method_table,file_name,'Sheet','Nabf','Range',row_name);
        writetable(method_table,file_name,'Sheet','SSIM','Range',row_name);
        writetable(method_table,file_name,'Sheet','MS_SSIM','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_pixel','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_dct','Range',row_name);
        writetable(method_table,file_name,'Sheet','FMI_w','Range',row_name);
        
    end
end
