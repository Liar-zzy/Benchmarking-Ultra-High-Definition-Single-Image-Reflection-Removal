import torch
import util.util as util
import models
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer
import numpy as np
import json
import glob
class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations
            

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))
            
            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)          

            self.iterations += 1
        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                    (self.epoch, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % 
                (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                (time.time() - epoch_start_time))
                
        # model.update_learning_rate()
        # train_loader.reset()

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):
        
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        # print(val_loader)
        with torch.no_grad():
            for i, data in enumerate(val_loader):                
                index = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                
                util.progress_bar(i, len(val_loader), str(avg_meters))
                
        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters


    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt

        if not os.path.exists(os.path.join(savedir,'json')):
            os.makedirs(os.path.join(savedir,'json'))

        psnr_results = []
        ssim_results = []
        results_json_list = []
        num = 0
        if opt.ctest == '1':
            path = join(savedir,'json')
            
            print('Continue Testing %s.'%path)
            for list in glob.glob(path+'/*.json'):
                with open(list,'r') as load_f:     
                    input_list = json.load(load_f)
                load_f.close()
                psnr_results.append(input_list['psnr'])
                ssim_results.append(input_list['ssim'])
                num = num + 1
            print("exist %d"%num)
            print("exist avg.psnr:",np.mean(psnr_results))
            print("exist avg.psnr:",np.mean(ssim_results))
            print("\n")
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                torch.cuda.empty_cache()
                if(i < num):
                    # print('continue %d'%i)
                    continue
                print('Process %d/%d .'%(i,len(test_loader)))
                # print(data)
                name, result_psnr, result_ssim = model.test(data, savedir=savedir)
                print('\n')
                util.progress_bar(i, len(test_loader))
                psnr_results.append(result_psnr)
                ssim_results.append(result_ssim)
                print('\n%d avg.psnr:%f | avg.ssim:%f'%(i,np.mean(psnr_results),np.mean(ssim_results)))

                per_result = {'filename':name, 'psnr':result_psnr,'ssim':result_ssim}
                results_json_list.append(per_result)

                json_path = os.path.join(savedir,'json','%04d.json'%i)
                
                with open(json_path,'w') as load_f:     
                    json.dump(per_result, load_f)
                load_f.close()
        
        print("Total psnr:",np.mean(psnr_results))
        print("Total ssim:",np.mean(ssim_results))
        output_results = []
        output_results.append({'results':results_json_list,'psnr_result':np.mean(psnr_results),'ssim_result':np.mean(ssim_results)})

        json_path = os.path.join(savedir,'psnr_ssim_results.json')
        with open(json_path,'w') as load_f:     
            json.dump(output_results, load_f)
        load_f.close()
    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
