################################################################################################
# Contain the compression code of DiSparse and other utility functions
################################################################################################

from copy import deepcopy
import torch.nn as nn
import torch
import types
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np

################################################################################################
# Overwrite PyTorch forward function for Conv2D and Linear to take the mask into account
# 覆盖Conv2D和Linear的PyTorch正向函数以考虑掩码
def hook_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def hook_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)






################################################################################################
# DiSparse prune in the static setup
# 静态设置中的DiSparse修剪                                     
# net: model to sparsify, should be of SceneNet class
# net：要稀疏化的模型，应为SceneNet类                    
# criterion: loss function to calculate per task gradients, should be of DiSparse_SceneNetLoss class
# criterion: 用于计算每个任务梯度的损失函数，应为DiSparse_SceneNetLoss类
# train_loader: dataloader to fetch data batches used to estimate importance
# train_loader: 数据加载器，用于获取用于估计重要性的数据批
# keep_ratio: how many parameters to keep
# keep_ratio: 要保留多少个参数
# tasks: set of tasks
# tasks: 任务集
def disparse_prune_static_channel(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
    # Register Hook
    # xavier的初始化方式和BN一样，为了保证数据的分布（均值方差一致）是一样的，加快收敛
    # torch.ones_like(layer.weight)生成与layer.weight形状相同、元素全为1的张量
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        # 通过 types.MethodType 函数，重写了卷积层和线性层的 forward 方法。
        # 这意味着原本的前向传播计算逻辑被替换为了自定义的 hook_forward_conv2d 和 hook_forward_linear 函数。
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    # Estimate importance per task in a data-driven manner
    # 以数据驱动的方式估计每个任务的重要性 
    # torch.cuda.empty_cache() 是 PyTorch 中的一个函数，用于清空 CUDA 设备（通常是 GPU）上的缓存内存。
    train_iter = iter(train_loader)
    for i in range(num_batches):

        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
        if "keypoint" in gt_batch:
            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
        if "edge" in gt_batch:
            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
        
        # 执行一个基于 PyTorch 的多任务学习场景中的模型测试过程。
        # 初始化预测变量 preds 为 None。
        # torch.cuda.empty_cache()清空 GPU 缓存以优化显存使用，确保有足够的空间运行下一次前向传播和反向传播。
        # 将测试网络（test_net）的所有梯度清零，这是在训练过程中必须的步骤，以防梯度累加到之前的状态上。
        # 对于当前任务的输入图像（gt_batch['img']），通过 test_net 进行前向传播计算得到预测结果 preds。
        # 计算损失函数 loss，这里假设损失函数 criterion 能够处理多任务，并且接受模型预测值、真实标签以及当前的任务索引 cur_task 作为输入。
        # 反向传播损失函数以计算所有可训练参数的梯度。
        # 初始化一个计数器 ct 用于记录遍历到的具有权重的层的数量。
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            # 如果模块是卷积层（nn.Conv2d）或全连接层（nn.Linear），并且该层属于 backbone 或者与当前任务 task[i+1] 相关（根据名称判断），
            # 更新或添加到一个名为 grads_abs[task] 的列表中，存储的是对应层权重掩码（weight_mask）的梯度绝对值。
            # 随着遍历的进行，计数器 ct 递增。
            # len(grads_abs[task]) > ct: 梯度更新时进行累计  ct最大48
            # 这段代码替换为需要的替换掉值 如 以folps的
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_mask.grad.data*layer.weight.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_mask.grad.data*layer.weight.data))
                        ct += 1
                  

    preds = None
    loss = None
    # Calculate Threshold
    # 计算阈值
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently 
    # 独立获取每个任务的重要性分数
    # 获取当前任务 task 对应的权重掩码梯度绝对值集合 cur_grads_abs。
    # 将所有梯度绝对值展开（flatten）后拼接成一个单一的张量 all_scores。
    # 计算所有梯度绝对值的总和作为归一化因子 norm_factor，然后将所有得分除以这个因子进行归一化处理。
    # 根据保留比例 keep_ratio 确定要保留的参数数量 num_params_to_keep，并从归一化后的梯度绝对值中找到最大的 num_params_to_keep 个得分及其对应的阈值 threshold。
    # 设定可接受的最低得分阈值为 acceptable_score，即第 num_params_to_keep 大的梯度绝对值。
    # 遍历当前任务的所有权重掩码梯度绝对值子集 cur_grads_abs，基于设定的 acceptable_score 创建一个新的二值掩码，表示哪些参数应当被保留（值为1）或舍弃（值为0）。这些二值掩码会被添加到 keep_masks[task] 列表中。
    # 最后打印出当前任务下被标记为保留（值为1）的参数总数。 
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]         # len(cur_grads_abs) = 48    cur_grads_abs[0].shape -->torch.Size([64, 3, 7, 7]) 以后的形状根据conv和linear层变化 cur_grads_abs[7].shape-->torch.Size([128, 64, 3, 3])
       # all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs]) # len(all_scores)=44500160  len(cur_grads_abs) = 48 
        all_chanel_scores =torch.sqrt(torch.cat([torch.sum(x**2,axis =(-3,-2,-1)) for x in cur_grads_abs]))
        #norm_factor = torch.sum(all_scores)     # 这一行计算all_scores张量中所有元素的总和，即梯度绝对值的L1范数。这个数值将被用作归一化因子。
        #all_scores.div_(norm_factor)            # 最后这一行是对all_scores张量进行原地除法操作，即直接修改all_scores的内容而不是创建新的张量。
        # norm_factor = torch.sum(all_chanel_scores)
        # all_chanel_scores.div_(norm_factor) 


        num_params_to_keep = int(len(all_chanel_scores) * keep_ratio)                   # 留下多少tensor
        threshold, _ = torch.topk(all_chanel_scores, num_params_to_keep, sorted=True)  # threshold会是一个一维张量,从大到小排序，包含num_params_to_keep个最大的分数；而未使用的第二个返回值（用下划线 _ 表示）则是对应这些最大分数的索引。  torch.topk()函数在PyTorch中用于从一个张量中找到指定数量（num_params_to_keep）的最大或最小值及其对应的索引。
        acceptable_score = threshold[-1]       # acceptable_score 阈值

        # 归一化后与阈值进行比较获得每层的剪枝的掩码keep_masks[task] 以 0 1的形式 
        # for g in cur_grads_abs:
        #     keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())
            # 创建和填充剪枝掩码
        for g in cur_grads_abs:
            # 根据通道重要性得分与阈值比较生成剪枝掩码  
            l2_norm = torch.sqrt(torch.sum(g**2,axis =(-3,-2,-1))).to(device)
            mask = []
            for m in l2_norm:
                if m >= acceptable_score:
                    mask.append(np.ones((g.shape[1], g.shape[2], g.shape[3])))
                else:
                    mask.append(np.zeros((g.shape[1], g.shape[2], g.shape[3])))
            mask = torch.from_numpy(np.array(mask),)
            keep_masks[task].append(mask.int()) # 使用uint8类型存储0/1掩码
        # 使用 torch.sum() 计算这个大的一维布尔型张量中 True 的个数 ---> x == 1 判断展平后的张量中每个元素是否等于1，结果会是一个布尔型张量
        # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    # Use PyTorch Prune to set hooks 
    # 使用PyTorch Prune设置钩子
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))       # 每一层的权重参数 'weight'

    # Use a prune ratio of 0 to set dummy pruning hooks 
    # 使用0的修剪比率设置虚拟修剪挂钩 这里的 amount=0 表示并没有真正地进行剪枝，而是设置了虚拟修剪挂钩。
    # 意味着虽然不立即移除任何参数，但已经为这些层注册了剪枝回调函数，之后可以更改这个 amount 参数来执行不同程度的剪枝操作，而无需重新设置剪枝策略。
    # 这种做法通常用于在训练过程中动态调整剪枝比率或在训练前后分阶段执行剪枝任务。
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )
    # prune.L1Norm()

    # Compute the final mask
    # 计算最终掩码
    # 创建了一个与 tasks 列表长度相等的整数列表，并将所有元素初始化为0。
    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection. 
            # The following is equivalent to elementwise OR by demorgan 
            # Only all tasks agree to prune, we prune 
            # 到达十字路口
            # 以下计算等效于通过demorgan的elementwise OR
            # 只有所有任务都同意修剪，我们修剪
            if 'backbone' in name:
                final_mask = None
                # 第一个任务的final_mask一定是 None所以去第一个if 
                # 剩下的final_mask一定不是None 所以去第一个else执行 逻辑与操作就得到所有任务都需要的掩码 
                # ~符号是Python中的按位取反运算符。  到所有任务都同意修剪的权重掩码。
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data                  #第一个任务同意修剪的权重掩码。
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)   #后续任务也都同意修剪的权重掩码。
                layer.weight_mask.data = ~final_mask    
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:             #得到 各自任务分别同意剪枝的掩码
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
                
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1

            else:
                print(f"Unrecognized Name: {name}!")
    # Forward
    for module in net.modules():
        # Check if it's basic block 
        # 检查它是否是basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig.to(device) * module.weight_mask.to(device)
            
    print_sparsity(net)
    return net






################################################################################################
# DiSparse prune in the static setup
# 静态设置中的DiSparse修剪                                     
# net: model to sparsify, should be of SceneNet class
# net：要稀疏化的模型，应为SceneNet类                    
# criterion: loss function to calculate per task gradients, should be of DiSparse_SceneNetLoss class
# criterion: 用于计算每个任务梯度的损失函数，应为DiSparse_SceneNetLoss类
# train_loader: dataloader to fetch data batches used to estimate importance
# train_loader: 数据加载器，用于获取用于估计重要性的数据批
# keep_ratio: how many parameters to keep
# keep_ratio: 要保留多少个参数
# tasks: set of tasks
# tasks: 任务集
def disparse_prune_static(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
    # Register Hook
    # xavier的初始化方式和BN一样，为了保证数据的分布（均值方差一致）是一样的，加快收敛
    # torch.ones_like(layer.weight)生成与layer.weight形状相同、元素全为1的张量
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        # 通过 types.MethodType 函数，重写了卷积层和线性层的 forward 方法。
        # 这意味着原本的前向传播计算逻辑被替换为了自定义的 hook_forward_conv2d 和 hook_forward_linear 函数。
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    # Estimate importance per task in a data-driven manner
    # 以数据驱动的方式估计每个任务的重要性 
    # torch.cuda.empty_cache() 是 PyTorch 中的一个函数，用于清空 CUDA 设备（通常是 GPU）上的缓存内存。
    train_iter = iter(train_loader)
    for b_idx in range(num_batches):

        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
        if "keypoint" in gt_batch:
            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
        if "edge" in gt_batch:
            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
        
        # 执行一个基于 PyTorch 的多任务学习场景中的模型测试过程。
        # 初始化预测变量 preds 为 None。
        # torch.cuda.empty_cache()清空 GPU 缓存以优化显存使用，确保有足够的空间运行下一次前向传播和反向传播。
        # 将测试网络（test_net）的所有梯度清零，这是在训练过程中必须的步骤，以防梯度累加到之前的状态上。
        # 对于当前任务的输入图像（gt_batch['img']），通过 test_net 进行前向传播计算得到预测结果 preds。
        # 计算损失函数 loss，这里假设损失函数 criterion 能够处理多任务，并且接受模型预测值、真实标签以及当前的任务索引 cur_task 作为输入。
        # 反向传播损失函数以计算所有可训练参数的梯度。
        # 初始化一个计数器 ct 用于记录遍历到的具有权重的层的数量。
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            #如果模块是卷积层（nn.Conv2d）或全连接层（nn.Linear），并且该层属于 backbone 或者与当前任务 task[i+1] 相关（根据名称判断），
            # 更新或添加到一个名为 grads_abs[task] 的列表中，存储的是对应层权重掩码（weight_mask）的梯度绝对值。
            # 随着遍历的进行，计数器 ct 递增。
            # len(grads_abs[task]) > ct: 梯度更新时进行累计  ct最大48
                    # 这段代码替换为需要的替换掉值 如 以folps的
            # for name, layer in test_net.named_modules():
            #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #         if 'backbone' in name or f'task{i+1}' in name:
            #             # 获取当前层的权重和权重掩码
            #             weight = layer.weight.data
            #             weight_mask = layer.weight_mask.grad.data

            #             # 初始化一个列表来存储每个通道的梯度绝对值
            #             channel_grads_abs = []

            #             # 遍历每个通道的权重，累积梯度绝对值
            #             for channel in weight:
            #                 channel_grad = torch.abs(channel * weight_mask).sum()
            #                 channel_grads_abs.append(channel_grad)

            #             # 更新任务的累积梯度列表
            #             if len(grads_abs[task]) > ct:
            #                 grads_abs[task][ct] += sum(channel_grads_abs)
            #             else:
            #                 grads_abs[task].append(sum(channel_grads_abs))

            #             # 更新计数器
            #             ct += 1
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_mask.grad.data*layer.weight.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_mask.grad.data*layer.weight.data))

                        if b_idx == (num_batches - 1):
                            grads_abs[task][ct] *= (layer.weight.data*layer.weight.data).pow(2)
                        ct += 1

    preds = None
    loss = None
    # Calculate Threshold
    # 计算阈值
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently 
    # 独立获取每个任务的重要性分数
    # 获取当前任务 task 对应的权重掩码梯度绝对值集合 cur_grads_abs。
    # 将所有梯度绝对值展开（flatten）后拼接成一个单一的张量 all_scores。
    # 计算所有梯度绝对值的总和作为归一化因子 norm_factor，然后将所有得分除以这个因子进行归一化处理。
    # 根据保留比例 keep_ratio 确定要保留的参数数量 num_params_to_keep，并从归一化后的梯度绝对值中找到最大的 num_params_to_keep 个得分及其对应的阈值 threshold。
    # 设定可接受的最低得分阈值为 acceptable_score，即第 num_params_to_keep 大的梯度绝对值。
    # 遍历当前任务的所有权重掩码梯度绝对值子集 cur_grads_abs，基于设定的 acceptable_score 创建一个新的二值掩码，表示哪些参数应当被保留（值为1）或舍弃（值为0）。这些二值掩码会被添加到 keep_masks[task] 列表中。
    # 最后打印出当前任务下被标记为保留（值为1）的参数总数。 
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]         # len(cur_grads_abs) = 48    cur_grads_abs[0].shape -->torch.Size([64, 3, 7, 7]) 以后的形状根据conv和linear层变化 cur_grads_abs[7].shape-->torch.Size([128, 64, 3, 3])
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs]) # len(all_scores)=44500160  len(cur_grads_abs) = 48 
        norm_factor = torch.sum(all_scores)     # 这一行计算all_scores张量中所有元素的总和，即梯度绝对值的L1范数。这个数值将被用作归一化因子。
        all_scores.div_(norm_factor)            # 最后这一行是对all_scores张量进行原地除法操作，即直接修改all_scores的内容而不是创建新的张量。

        num_params_to_keep = int(len(all_scores) * keep_ratio)                   # 留下多少tensor
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)  # threshold会是一个一维张量,从大到小排序，包含num_params_to_keep个最大的分数；而未使用的第二个返回值（用下划线 _ 表示）则是对应这些最大分数的索引。  torch.topk()函数在PyTorch中用于从一个张量中找到指定数量（num_params_to_keep）的最大或最小值及其对应的索引。
        acceptable_score = threshold[-1]       # acceptable_score 阈值

        # 归一化后与阈值进行比较获得每层的剪枝的掩码keep_masks[task] 以 0 1的形式 
        for g in cur_grads_abs:
            keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())
        # 使用 torch.sum() 计算这个大的一维布尔型张量中 True 的个数 ---> x == 1 判断展平后的张量中每个元素是否等于1，结果会是一个布尔型张量
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    # Use PyTorch Prune to set hooks 
    # 使用PyTorch Prune设置钩子
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))       # 每一层的权重参数 'weight'

    # Use a prune ratio of 0 to set dummy pruning hooks 
    # 使用0的修剪比率设置虚拟修剪挂钩 这里的 amount=0 表示并没有真正地进行剪枝，而是设置了虚拟修剪挂钩。
    # 意味着虽然不立即移除任何参数，但已经为这些层注册了剪枝回调函数，之后可以更改这个 amount 参数来执行不同程度的剪枝操作，而无需重新设置剪枝策略。
    # 这种做法通常用于在训练过程中动态调整剪枝比率或在训练前后分阶段执行剪枝任务。
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    # Compute the final mask
    # 计算最终掩码
    # 创建了一个与 tasks 列表长度相等的整数列表，并将所有元素初始化为0。
    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection. 
            # The following is equivalent to elementwise OR by demorgan 
            # Only all tasks agree to prune, we prune 
            # 到达十字路口
            # 以下计算等效于通过demorgan的elementwise OR
            # 只有所有任务都同意修剪，我们修剪
            if 'backbone' in name:
                final_mask = None
                # 第一个任务的final_mask一定是 None所以去第一个if 
                # 剩下的final_mask一定不是None 所以去第一个else执行 逻辑与操作就得到所有任务都需要的掩码 
                # ~符号是Python中的按位取反运算符。  到所有任务都同意修剪的权重掩码。
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data                  #第一个任务同意修剪的权重掩码。
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)   #后续任务也都同意修剪的权重掩码。
                layer.weight_mask.data = ~final_mask    
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:             #得到 各自任务分别同意剪枝的掩码
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
                
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1

            else:
                print(f"Unrecognized Name: {name}!")
    # Forward
    for module in net.modules():
        # Check if it's basic block 
        # 检查它是否是basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
            
    print_sparsity(net)
    return net

################################################################################################
# DiSparse prune in the pretrained setup
# net: model to sparsify, should be a SceneNet class
# criterion: loss function to calculate per task gradients, should be DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def disparse_prune_pretrained(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []

    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    for b_idx in range(num_batches):
        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
            
        
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight.data*layer.weight.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight.data*layer.weight.grad.data))
                        
                        if b_idx == (num_batches - 1):
                            grads_abs[task][ct] *= (layer.weight.data*layer.weight.data).pow(2)
                            
                        ct += 1
                              
    preds = None
    loss = None
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
 
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for g in cur_grads_abs:
            keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    

    # Use PyTorch Prune to set hooks
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    # Use a prune ratio of 0 to set dummy pruning hooks
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    # Compute the final mask
    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)
                layer.weight_mask.data = ~final_mask
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1
            else:
                print(f"Unrecognized Name: {name}!")
    # Forward
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
            
    print_sparsity(net)
    return net

################################################################################################
# DiSparse prune in the pretrained setup using the l1 score
# net: model to sparsify, should be a SceneNet class
# criterion: loss function to calculate per task gradients, should be DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def disparse_prune_pretrained_l1(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    mag_abs = {}
    for task in tasks:
        mag_abs[task] = []
        
    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    idxs = [0] * len(tasks)
    ct = 0
    
    for i, task in enumerate(tasks):
        for name, layer in test_net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if 'backbone' in name or f'task{i+1}' in name:
                    if len(mag_abs[task]) > ct:
                        mag_abs[task][ct] += torch.abs(layer.weight.data*layer.weight.grad.data)
                    else:
                        mag_abs[task].append(torch.abs(layer.weight.data*layer.weight.grad.data))
                    ct += 1
    
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_mag_abs = mag_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_mag_abs])
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        for g in cur_mag_abs:
            keep_masks[task].append(((g) >= acceptable_score).int())
            
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)
                layer.weight_mask.data = ~final_mask
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
                
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1
                
            else:
                print(f"Unrecognized Name: {name}!")
                
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    print_sparsity(net)
    
    return net

################################################################################################
def print_sparsity(prune_net, printing=True):
    # Prine the sparsity
    num = 0
    denom = 0
    ct = 0
    for module in prune_net.modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            if hasattr(module, 'weight'):
                num += torch.sum(module.weight == 0)
                denom += module.weight.nelement()
                if printing:
                    print(
                    f"Layer {ct}", "Sparsity in weight: {:.2f}%".format(
                        100. * torch.sum(module.weight == 0) / module.weight.nelement())
                    )
                ct += 1
    if printing:
        print(f"Model Sparsity Now: {num / denom * 100}")
    return num / denom

################################################################################################
def get_pruned_init(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module = prune.identity(module, 'weight')
    return net

################################################################################################
def deepcopy_pruned_net(net, copy_net):
    copy_net = get_pruned_init(copy_net)
    copy_net.load_state_dict(net.state_dict())
    return copy_net

################################################################################################
def get_sparsity_dict(net):
    sparsity_dict = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            if hasattr(module, 'weight'):
                sparsity_dict[name] = torch.sum(module.weight == 0) / module.weight.nelement()
                sparsity_dict[name] = sparsity_dict[name].item()
    return sparsity_dict

################################################################################################
def pseudo_forward(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    return net

################################################################################################