# General imports
import os
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:

class VisDispatcher:
    def __init__(self,act_pool_procs,filt_pool_procs):
        self.act_pool_procs = act_pool_procs
        self.filt_pool_procs = filt_pool_procs
        self.manager = multiprocessing.Manager()
        self.activations_vis_queue = self.manager.Queue()
        self.filters_vis_queue = self.manager.Queue()
        self.activations_work_pool = multiprocessing.Pool(self.act_pool_procs, self.vis_activations_worker_small, (self.activations_vis_queue,))
        self.filters_work_pool = multiprocessing.Pool(self.filt_pool_procs, self.vis_filters_worker, (self.filters_vis_queue,))
    
    @staticmethod
    def plot_2D_twoslope_image(array2D,save_path,dpi =600):
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        act_min = array2D.min()
        act_max = array2D.max()
        center = 0
        if (act_min < 0) and (act_max < 0):
            act_max = 1
        if (act_min > 0) and (act_max > 0):
            act_min = -1


        divnorm = colors.TwoSlopeNorm(vmin=act_min,vcenter=0,vmax=act_max)
        im = ax1.imshow(array2D,interpolation=None,cmap="seismic",norm=divnorm)
        fig1.colorbar(im, shrink=0.8, extend='both', label='Color values')
        fig1.savefig(save_path, dpi=dpi)
        
        plt.close(fig1)
    
    @staticmethod
    def plot_histogram(flatt_array,save_path,dpi =200):
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.hist(flatt_array, 40, rwidth=0.8)
        fig1.savefig(save_path,dpi=dpi)
        plt.close(fig1)
    
    @staticmethod
    def vis_activations_worker(q):
        while True:
            item = q.get(True)
            if item is None:
                break

            epoch = item["epoch"]
            batch_files = item["batch_files"]
            activations = item["activations"]
            
            for b in range(activations.shape[0]):
                sample_name = f"{batch_files[b].split('/')[-2].replace('.','_')}"
                save_path = f"/home/erikj/projects/insidrug/py_proj/erikj/log_nn_vis/{epoch}/{sample_name}/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for filt in range(activations.shape[1]):
                    filt_activation = activations[b][filt]
                    fig1, ax1 = plt.subplots(figsize=(5, 5))
                    divnorm = colors.TwoSlopeNorm(vmin=filt_activation.min(),vcenter=0,vmax=filt_activation.max())
                    ax1.imshow(filt_activation,interpolation=None,cmap="seismic",norm=divnorm)
                    fig1.savefig(f"{save_path}filt_{filt}.png", dpi=400)
                
                plt.close(fig1)
    
    @staticmethod
    def vis_activations_worker_small(q):
        while True:
            item = q.get(True)
            if item is None:
                break
            
            activation = item["activation"]
            filename = item["filename"]
            VisDispatcher.plot_2D_twoslope_image(activation,filename)
            # VisDispatcher.plot_histogram(activation.flatten(),)

    
    @staticmethod
    def vis_filters_worker(q):
        while True:
            item = q.get(True)
            if item is None:
                break
            t = item["type"]
            if t == "image":
                filt = item["filter"]
                save_path = item["save_path"]
                VisDispatcher.plot_2D_twoslope_image(filt,save_path,dpi=100)
            elif t == "histogram":
                filters = item["filters"]
                save_path = item["save_path"]
                filters = filters.flatten()
                VisDispatcher.plot_histogram(filters,save_path)

    
    def terminate_dispatcher(self):
        for x in range(self.act_pool_procs):
            print(f"Closing worker {x}")
            self.activations_vis_queue.put(None)
        self.activations_work_pool.close()
        self.activations_work_pool.join()
        return self

    





if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")