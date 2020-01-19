close all;
clear all;

datasets = {'avenue', 'ped1', 'ped2', 'enter', 'exit'};

cost_files_dir = 'recon_costs\';
save_path = 'C:\Users\admin\MATLAB-workspace\all_dataset_post_processing\plots_with_regs\';

for d = 1:1
    dataset = datasets{d};
    fprintf('Dataset: %s\n', dataset);
    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    if strcmp(dataset, 'avenue')
        load('gt_avenue.mat');
        threshold = 0.2;
        start_video = 1;
        num_video = 21;
        stride = 5;
        h = figure(1);
        set(h,'Position',[2000, 300,700,200]);
    elseif strcmp(dataset, 'ped1')
        load('gt_ped1.mat');
        threshold = 0.3;
        start_video = 1;
        num_video = 36;
        stride = 5;
        h = figure(1);
        set(h,'Position',[2000,100,400,200]);
    elseif strcmp(dataset, 'ped2')
        load('gt_ped2.mat');
        threshold = 0.3;
        start_video = 1;
        num_video = 12;
        stride = 5;
        h = figure(1);
        set(h,'Position',[2000,100,400,200]);
    elseif strcmp(dataset, 'enter')
        load('gt1_enter.mat');
        threshold = 0.2;
        start_video = 1;
        num_video = 5;
        stride = 10;
        h = figure(1);
        set(h,'Position',[2000,100,2500,200]);
    elseif strcmp(dataset, 'exit')
        load('gt1_exit.mat');
        threshold = 0.25;
        start_video = 1;
        num_video = 4;
        stride = 10;
        h = figure(1);
        set(h,'Position',[2000,100,2000,200]);
    end

    for i = start_video:start_video+num_video-1
        agt = gt{i};

        cost_file_name = sprintf('%s_video_%02d_conv3_iter_150000.txt',dataset,i);
        cost_file = sprintf('%s%s',cost_files_dir,cost_file_name);
        fprintf('%s\n', cost_file);

        % computing regularity
        data = importdata(cost_file);
        data = data(data>0);
        ndata = imresize(data,[stride*size(data,1),1]);
        ndata = ndata-min(ndata);
        ndata = 1-ndata/max(ndata);
        % ndata = medfilt1(ndata, 20);

        % thresholding
        [minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = ...
            run_persistence1d(single(ndata)); 
        persistent_features =  filter_features_by_persistence(minIndices, maxIndices, persistence, threshold); 
        minima_indices = [persistent_features(:,1); globalMinIndex];
        
        l1 = plot(ndata,'LineWidth',2, 'Color', 'b');
        hold on;
      
        markers = ndata(minima_indices);
        s1 = scatter(minima_indices, markers, 100, 'm', 'fill');
        
        min_cost = 0;
        max_cost = 1;
        
        for k = 1:size(agt,2)
            sframe = agt(1,k);
            eframe = agt(2,k);
            p1 = patch([sframe,sframe:eframe,eframe],[min_cost,max_cost*ones(1,eframe-sframe+1),min_cost],'r');
            set(p1,'FaceAlpha',0.3,'EdgeColor','r');       
        end
        
        abnormal_regs = combine_locals(length(ndata), minima_indices, 100);
        abnormal_regs = abnormal_regs';
        for k = 1:size(abnormal_regs,2)
            sframe = abnormal_regs(1,k);
            eframe = abnormal_regs(2,k);
            p2 = patch([sframe,sframe:eframe,eframe],[min_cost,max_cost*ones(1,eframe-sframe+1),min_cost],'g');
            set(p2,'FaceAlpha',0.3,'EdgeColor','g');       
        end
        
        [det, gtg] = compute_overlaps(abnormal_regs, agt);
        tp = tp + sum(det==1);
        fp = fp + sum(det==0);
        fn = fn + sum(gtg==0);
        tn = tn + sum(gtg==1);
        
        xlim([0,length(ndata)]);
        set(gca,'FontSize',14);
        legend([l1,s1,p1,p2],'Generalized Model','Local Mimimas','Ground Truth','Detection','Location','southoutside','Orientation','horizontal');
        hold off;
        % export_fig(sprintf('%s/%s_video_%02d.png',save_path,dataset,i),'-transparent');
    end
    fprintf('TP: %d\n', tp);
    fprintf('FP: %d\n', fp);
    fprintf('FN: %d\n', fn);
    fprintf('TN: %d\n', tn);
    fprintf('Precision: %0.2f\n', tp/(tp+fp));
    fprintf('Recall: %0.2f\n', tp/(tp+fn));
    fprintf('True Positive Rate: %0.2f\n', tp/(tp+fn));
    fprintf('False Positive Rate: %0.2f\n', fp/(tn+fp));
end

