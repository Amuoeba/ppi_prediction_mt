# General imports
import os
# Project specific imports

# Imports from internal libraries
from nn_utils.layer_vis import VisDispatcher
import utils
# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:


# def verbose_layer(self, input, output, **kwargs):
#     # input is a tuple of packed inputs
#     # output is a Tensor. output.data is the Tensor we are interested
#     print("-"*30)
#     print(f"PID: {os.getpid()}")
#     print(
#         f"Layer: {kwargs['layer_name']}, {self.__class__.__name__} | FORWARD")
#     print(
#         f"Input size: {str(input[0].size()):^35}  |  min: {input[0].min():6.4f}  |  max: {input[0].max():6.4f}")
#     print(
#         f"output size:{str(output.data.size()):^35}  |  min: {output.data.min():6.4f}  |  max: {output.data.max():6.4f}")
#     print("-"*60)


class HookManager:
    def __init__(self, network, hookable, verbose=False):
        self.network = network
        self.hookable = hookable
        self.hooks = []
        self.verbose = verbose

    @staticmethod
    def hook_function(m, input, output, **kwargs):
        print("-" * 30)
        print(f"PID: {os.getpid()}")
        print(
            f"Layer: {kwargs['layer_name']}, {m.__class__.__name__} | FORWARD")
        print(
            f"Input size: {str(input[0].size()):^35}  |  min: {input[0].min():6.4f}  |  max: {input[0].max():6.4f}")
        print(
            f"output size:{str(output.data.size()):^35}  |  min: {output.data.min():6.4f}  |  max: {output.data.max():6.4f}")
        print("-" * 60)

    def _register_hooks_(self, modules, parents=[]):
        if isinstance(modules, list):
            for m in modules:
                self._register_hooks_(m, parents)
        if isinstance(modules, str):
            if self.verbose:
                print(f"Hooking: {modules} Parents: {parents}")
            module = self.network
            for p in parents:
                module = module._modules[p]
            hook = module._modules[modules].register_forward_hook(
                lambda s, i, o: self.hook_function(s, i, o, layer_name=f"{'.'.join(parents)}.{modules}"))
            self.hooks.append(hook)
        if isinstance(modules, dict):
            for m in modules:
                aux_parents = parents.copy()
                aux_parents.append(m)
                self._register_hooks_(modules[m], aux_parents)

    def register_hooks(self):
        self._register_hooks_(self.hookable)

    def unregister_hooks(self):
        for hook in self.hooks:
            if self.verbose:
                print(f"Removing hook:{hook}")
            hook.remove()
        self.hooks = []


class VisHookManager:
    def __init__(self, network, visualizer: VisDispatcher,
                 activation_vis_hookable=None,
                 activation_histogram_hooks=None,
                 filter_vis_hookable=None,
                 verbose=False):
        self.network = network
        self.verbose = verbose
        self.activation_vis_hookable = activation_vis_hookable
        self.filter_vis_hookable = filter_vis_hookable

        self.activation_vis_hooks = []
        self.filter_vis_hooks = []

        self.visualizer = visualizer

        self.epoch = None
        self.batch_files = []

    def set_run_metadata(self, epoch, batch_file_names):
        self.epoch = epoch
        self.batch_files = batch_file_names
        return self

    ###################################################################################################
    # Activation 2D image hooks
    ###################################################################################################
    @staticmethod
    def act_vis_hook_fun(m, input, output, **kwargs):
        layer = kwargs["layer_name"]
        visualizer = kwargs["visualizer"]
        epoch = kwargs["epoch"]
        batch_files = kwargs["batch_files"]
        weights = m.weight
        activations = output.detach().clone().cpu().numpy()

        for b in range(activations.shape[0]):
            sample_name = f"{batch_files[b].split('/')[-2].replace('.', '_')}"

            save_path = f"{utils.logger.nn_vis_path}/{epoch}/{sample_name}/{layer}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for filt in range(activations.shape[1]):
                filt_activation = activations[b][filt]
                visualizer.activations_vis_queue.put(
                    {"activation": filt_activation, "filename": f"{save_path}filt_{filt}.png"})

    def register_activation_hooks(self):
        self._register_activation_hooks_(self.activation_vis_hookable)

    def _register_activation_hooks_(self, modules, parents=[]):
        if isinstance(modules, list):
            for m in modules:
                self._register_activation_hooks_(m, parents)
        if isinstance(modules, str):
            if self.verbose:
                print(f"Hooking: {modules} Parents: {parents}")
            module = self.network
            for p in parents:
                module = module._modules[p]
            hook = module._modules[modules].register_forward_hook(
                lambda s, i, o: self.act_vis_hook_fun(s, i, o,
                                                      layer_name=f"{'.'.join(parents)}.{modules}",
                                                      visualizer=self.visualizer,
                                                      epoch=self.epoch,
                                                      batch_files=self.batch_files))
            self.activation_vis_hooks.append(hook)
        if isinstance(modules, dict):
            for m in modules:
                aux_parents = parents.copy()
                aux_parents.append(m)
                self._register_activation_hooks_(modules[m], aux_parents)

    def unregister_activation_vis_hooks(self):
        for hook in self.activation_vis_hooks:
            if self.verbose:
                print(f"Removing hook:{hook}")
            hook.remove()
        self.hooks = []

    ###################################################################################################
    # Activation histogram hooks
    ###################################################################################################

    # TODO Continuue here
    @staticmethod
    def activation_hist_vis_fun(m, input, output, **kwargs):
        layer = kwargs["layer_name"]
        visualizer = kwargs["visualizer"]
        epoch = kwargs["epoch"]
        batch_files = kwargs["batch_files"]
        weights = m.weight
        activations = output.detach().clone().cpu().numpy()

        for b in range(activations.shape[0]):
            sample_name = f"{batch_files[b].split('/')[-2].replace('.', '_')}"

            save_path = f"{utils.logger.nn_vis_path}/{epoch}/{sample_name}/{layer}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for filt in range(activations.shape[1]):
                filt_activation = activations[b][filt]
                visualizer.activations_vis_queue.put(
                    {"activation": filt_activation, "filename": f"{save_path}filt_histogram_{filt}.png"})

    def register_activation_histogram_hooks(self):
        self._register_activation_hooks_(self.activation_vis_hookable)

    def _register_activation_histogram_hooks_(self, modules, parents=[]):
        if isinstance(modules, list):
            for m in modules:
                self._register_activation_hooks_(m, parents)
        if isinstance(modules, str):
            if self.verbose:
                print(f"Hooking: {modules} Parents: {parents}")
            module = self.network
            for p in parents:
                module = module._modules[p]
            hook = module._modules[modules].register_forward_hook(
                lambda s, i, o: self.act_vis_hook_fun(s, i, o,
                                                      layer_name=f"{'.'.join(parents)}.{modules}",
                                                      visualizer=self.visualizer,
                                                      epoch=self.epoch,
                                                      batch_files=self.batch_files))
            self.activation_vis_hooks.append(hook)
        if isinstance(modules, dict):
            for m in modules:
                aux_parents = parents.copy()
                aux_parents.append(m)
                self._register_activation_hooks_(modules[m], aux_parents)

    ###################################################################################################
    # Filter vis hooks
    ###################################################################################################
    @staticmethod
    def filter_vis_hook_fun(m, input, output, **kwargs):
        layer = kwargs["layer_name"]
        visualizer = kwargs["visualizer"]
        epoch = kwargs["epoch"]
        batch_files = kwargs["batch_files"]
        filters = m.weight.detach().clone().cpu().numpy()

        filter_save_path = f"{utils.logger.nn_vis_path}/{epoch}/filter_viss/{layer}/"
        if not os.path.exists(filter_save_path):
            os.makedirs(filter_save_path)

        visualizer.filters_vis_queue.put(
            {"filters": filters, "save_path": f"{filter_save_path}weight_distributions.png", "type": "histogram"})

        # Uncomment this to visualize filters as 2d images
        # for x_next in range(filters.shape[0]):
        #     for x_prev in range(filters.shape[1]):
        #         visualizer.filters_vis_queue.put(
        #             {"filter": filters[x_next][x_prev], "save_path": f"{filter_save_path}prev_{x_next}_next_{x_prev}.png", "type": "image"})

    def register_filter_hooks(self):
        self._register_filter_hooks_(self.filter_vis_hookable)

    def _register_filter_hooks_(self, modules, parents=[]):
        if isinstance(modules, list):
            for m in modules:
                self._register_filter_hooks_(m, parents)
        if isinstance(modules, str):
            if self.verbose:
                print(f"Hooking: {modules} Parents: {parents}")
            module = self.network
            for p in parents:
                module = module._modules[p]
            hook = module._modules[modules].register_forward_hook(
                lambda s, i, o: self.filter_vis_hook_fun(s, i, o,
                                                         layer_name=f"{'.'.join(parents)}.{modules}",
                                                         visualizer=self.visualizer,
                                                         epoch=self.epoch,
                                                         batch_files=self.batch_files))
            self.activation_vis_hooks.append(hook)
        if isinstance(modules, dict):
            for m in modules:
                aux_parents = parents.copy()
                aux_parents.append(m)
                self._register_filter_hooks_(modules[m], aux_parents)

    def unregister_filter_vis_hooks(self):
        for hook in self.filter_vis_hooks:
            if self.verbose:
                print(f"Removing hook:{hook}")
            hook.remove()
        self.hooks = []


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
