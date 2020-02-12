import mxnet as mx
# import negativemining
# import negativemining_landmark
# import negativemining_onlylandmark
# import negativemining_onlylandmark10
# import negativemining_onlylandmark106
from config import config

#def P_Net16_v0(mode='train'):
def P_Net16(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 16 x 16
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2),  num_filter=16, name="conv1")#16/7
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#7/3
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#3/1
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net16_v1(mode='train'):
#def P_Net16(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 16 x 16
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#16/15
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#15/7
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#7/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v00(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1")#20/9
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#9/4
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(2, 2),num_filter=16, num_group=16, name="conv3_dw")#4/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

#def P_Net20_v0(mode='train'):
def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1")#20/9
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw")#9/7
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#7/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v1(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=8, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=16, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=16, num_group=16, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v2(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=24, num_group=24, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v3(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=8, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=24, num_group=24, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v4(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2,2), num_filter=32, num_group=32, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=32, num_group=32, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v5(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2,2), num_filter=32, num_group=32, name="conv4_dw")#7/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v6(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, num_group=32, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=48, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2,2), num_filter=48, num_group=48, name="conv4_dw")#7/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v7(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(2, 2), num_filter=24, num_group=24, name="conv3_dw")#9/8
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    pool2 = mx.symbol.Pooling(data=prelu3, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool2") #8/4
    conv4_sep = mx.symbol.Convolution(data=pool2, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(2, 2), num_filter=64, num_group=64, name="conv5_dw")#4/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")

    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v8(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=32, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=32, num_group=32, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=32, num_group=32, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=64, num_group=64, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_s2v1(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=16, num_group=16, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=16, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=16, num_group=16, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=32, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=32, num_group=32, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_s2v2(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=16, num_group=16, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_s2v3(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=32, num_group=32, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=48, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=48, num_group=48, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
#def R_Net_v1(mode='train'):
def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=32, name="conv1") #24/23
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw")#23/11
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw")#11/5
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_dw")#5/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    conv5_2 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group
	
def R_Net_v2(mode='train'):
#def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1") #24/22
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=32, num_group=32, name="conv2_dw")#22/21
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw")#21/10
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=32, num_group=32, name="conv4_dw")#10/9
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw")#9/4
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(2, 2), num_filter=64, num_group=64, name="conv6_dw")#4/3
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=128, name="conv6_sep")
    prelu6 = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6")

    conv7_dw = mx.symbol.Convolution(data=prelu6, kernel=(3, 3), num_filter=128, num_group=128, name="conv7_dw")#3/1
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu7_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    conv5_2 = mx.symbol.FullyConnected(data=prelu7_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group

#def O_Net_v1(mode="train", with_landmark = False):
def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=32, name="conv1") #48/47
        prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw") #47/23
        prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
        prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw") #23/11
        prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
        prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw") #11/5
        prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
        prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw") #5/3
        prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep")
        prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_dw") #3/1
        prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1,  name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

def O_Net_v2(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=32, name="conv1") #48/47
        prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw") #47/23
        prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep")
        prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw") #23/11
        prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
        prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw") #11/5
        prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
        prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw") #5/3
        prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep")
        prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw") #3/1
        prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1,  name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group
	
def O_Net_v3(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),num_filter=32, name="conv1", no_bias=True) #48/46
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True) #46/45
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True) #45/22
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True) #22/21
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw", no_bias=True) #21/10
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(2, 2), num_filter=64, num_group=64, name="conv6_dw", no_bias=True) #10/9
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
        conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=64, name="conv6_sep", no_bias=True)
        bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
        prelu6 = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6")
		
        conv7_dw = mx.symbol.Convolution(data=prelu6, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv7_dw", no_bias=True) #9/4
        bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
        prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
        conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=128, name="conv7_sep", no_bias=True)
        bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
        prelu7 = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7")
		
        conv8_dw = mx.symbol.Convolution(data=prelu7, kernel=(2, 2), num_filter=128, num_group=128, name="conv8_dw", no_bias=True) #4/3
        bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
        prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
        conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=256, name="conv8_sep", no_bias=True)
        bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
        prelu8 = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8")

        conv9_dw = mx.symbol.Convolution(data=prelu8, kernel=(3, 3), num_filter=256, num_group=256, name="conv9_dw", no_bias=True) #3/1
        bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
        prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu9_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu9_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

lnet_basenum=32
#def L_Net_v1(mode="train"):
def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1") #48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #46/45
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv3_dw") #45/22
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv4_dw") #22/21
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv5_dw") #21/10
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv6_dw") #10/9
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv7_dw") #9/4
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv8_dw") #4/3
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv9_dw") #3/1
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group

def L_Net_v2(mode="train"):	
#def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1") #48/46
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #46/45
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv3_dw") #45/22
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv4_dw") #22/21
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv5_dw") #21/10
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv6_dw") #10/9
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv7_dw") #9/4
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv8_dw") #4/3
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv9_dw") #3/1
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group
	
#def L_Net64_v3(mode="train"):
def L_Net64(mode="train"):
    """
    Refine Network
    input shape 3 x 64 x 64
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet_basenum, name="conv1") #64/63
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #63/31
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv3_dw") #31/15
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*4, num_group=lnet_basenum*4, name="conv4_dw") #15/7
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*4, num_group=lnet_basenum*4, name="conv5_dw") #7/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")

    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu6_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group
	
lnet106_basenum=32
#def L106_Net_v1(mode="train"):
def L106_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #46/45
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #45/22
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #22/21
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #21/10
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv6_dw") #10/9
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv7_dw") #9/4
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv8_dw") #4/3
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv9_dw") #3/1
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def L106_Net_v2(mode="train"):
#def L106_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #48/46
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #46/45
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #45/22
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #22/21
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #21/10
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv6_dw") #10/9
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv7_dw") #9/4
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv8_dw") #4/3
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv9_dw") #3/1
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
#def L106_Net96_v1(mode="train"):
def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #96/94
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #94/93
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #93/46
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #46/45
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #45/22
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv6_dw") #22/21
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv7_dw") #21/10
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")
	
    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv8_dw") #10/9
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv9_dw") #9/4
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv10_dw = mx.symbol.Convolution(data=prelu9_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv10_dw") #4/3
    prelu10_dw = mx.symbol.LeakyReLU(data=conv10_dw, act_type="prelu", name="prelu10_dw")
    conv10_sep = mx.symbol.Convolution(data=prelu10_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv10_sep")
    prelu10_sep = mx.symbol.LeakyReLU(data=conv10_sep, act_type="prelu", name="prelu10_sep")

    conv11_dw = mx.symbol.Convolution(data=prelu10_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv11_dw") #3/1
    prelu11_dw = mx.symbol.LeakyReLU(data=conv11_dw, act_type="prelu", name="prelu11_dw")
    conv11_sep = mx.symbol.Convolution(data=prelu11_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv11_sep")
    prelu11_sep = mx.symbol.LeakyReLU(data=conv11_sep, act_type="prelu", name="prelu11_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu11_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def L106_Net96_v2(mode="train"):
#def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #96/94
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #94/93
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #93/46
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #46/45
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #45/22
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv6_dw") #22/21
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv7_dw") #21/10
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")
	
    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv8_dw") #10/9
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")
	
    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv9_dw") #9/4
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv10_dw = mx.symbol.Convolution(data=prelu9_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv10_dw") #4/3
    bn10_dw = mx.sym.BatchNorm(data=conv10_dw, name='bn10_dw', fix_gamma=False,momentum=0.9)
    prelu10_dw = mx.symbol.LeakyReLU(data=bn10_dw, act_type="prelu", name="prelu10_dw")
    conv10_sep = mx.symbol.Convolution(data=prelu10_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv10_sep")
    bn10_sep = mx.sym.BatchNorm(data=conv10_sep, name='bn10_sep', fix_gamma=False,momentum=0.9)
    prelu10_sep = mx.symbol.LeakyReLU(data=bn10_sep, act_type="prelu", name="prelu10_sep")

    conv11_dw = mx.symbol.Convolution(data=prelu10_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv11_dw") #3/1
    bn11_dw = mx.sym.BatchNorm(data=conv11_dw, name='bn11_dw', fix_gamma=False,momentum=0.9)
    prelu11_dw = mx.symbol.LeakyReLU(data=bn11_dw, act_type="prelu", name="prelu11_dw")
    conv11_sep = mx.symbol.Convolution(data=prelu11_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv11_sep")
    bn11_sep = mx.sym.BatchNorm(data=conv11_sep, name='bn11_sep', fix_gamma=False,momentum=0.9)
    prelu11_sep = mx.symbol.LeakyReLU(data=bn11_sep, act_type="prelu", name="prelu11_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu11_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
	
def L106_Net96_v3(mode="train"):
#def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet106_basenum, name="conv1") #96/95
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #95/47
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv3_dw") #47/23
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv4_dw") #23/11
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv5_dw") #11/5
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv6_dw") #5/3
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv7_dw") #3/1
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu7_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
#def L106_Net64_v3(mode="train"):
def L106_Net64(mode="train"):
    """
    Refine Network
    input shape 3 x 64 x 64
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet106_basenum, name="conv1") #64/63
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #63/31
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv3_dw") #31/15
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv4_dw") #15/7
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv5_dw") #7/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")

    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu6_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group


def L106_Net_112(mode="train"):
    # def L106_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")

    conv1_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=8, num_group=1, pad=(1, 1),stride=(2, 2), data=data, name="conv1_conv2d")
    conv1_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=conv1_conv2d, name="conv1_batchnorm")
    conv1_relu = mx.symbol.LeakyReLU(act_type="prelu", data=conv1_batchnorm, name="conv1_relu")

    res2_block0_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=8, num_group=1,pad=(0, 0), stride=(1, 1), data=conv1_relu,name="res2_block0_conv_sep_conv2d")
    res2_block0_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res2_block0_conv_sep_conv2d,name="res2_block0_conv_sep_batchnorm")
    res2_block0_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res2_block0_conv_sep_batchnorm,name="res2_block0_conv_sep_relu")
    res2_block0_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=8, num_group=8,pad=(1, 1), stride=(1, 1), data=res2_block0_conv_sep_relu,name="res2_block0_conv_dw_conv2d")
    res2_block0_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res2_block0_conv_dw_conv2d,name="res2_block0_conv_dw_batchnorm")
    res2_block0_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res2_block0_conv_dw_batchnorm,name="res2_block0_conv_dw_relu")
    res2_block0_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=8, num_group=1,pad=(0, 0), stride=(1, 1), data=res2_block0_conv_dw_relu,name="res2_block0_conv_proj_conv2d")
    res2_block0_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res2_block0_conv_proj_conv2d,name="res2_block0_conv_proj_batchnorm")

    _plus0 = mx.symbol.elemwise_add(lhs=conv1_relu,rhs=res2_block0_conv_proj_batchnorm, name="_plus0")

    dconv23_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1, pad=(0, 0),stride=(1, 1), data=_plus0, name="dconv23_conv_sep_conv2d")
    dconv23_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv23_conv_sep_conv2d,name="dconv23_conv_sep_batchnorm")
    dconv23_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv23_conv_sep_batchnorm,name="dconv23_conv_sep_relu")
    dconv23_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=16, num_group=16, pad=(1, 1),stride=(2, 2), data=dconv23_conv_sep_relu,name="dconv23_conv_dw_conv2d")
    dconv23_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv23_conv_dw_conv2d,name="dconv23_conv_dw_batchnorm")
    dconv23_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv23_conv_dw_batchnorm,name="dconv23_conv_dw_relu")
    dconv23_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1,pad=(0, 0), stride=(1, 1), data=dconv23_conv_dw_relu,name="dconv23_conv_proj_conv2d")
    dconv23_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv23_conv_proj_conv2d,name="dconv23_conv_proj_batchnorm")

    res3_block0_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1,pad=(0, 0), stride=(1, 1), data=dconv23_conv_proj_batchnorm,name="res3_block0_conv_sep_conv2d")
    res3_block0_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res3_block0_conv_sep_conv2d,name="res3_block0_conv_sep_batchnorm")
    res3_block0_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res3_block0_conv_sep_batchnorm,name="res3_block0_conv_sep_relu")
    res3_block0_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=16, num_group=16,pad=(1, 1), stride=(1, 1), data=res3_block0_conv_sep_relu,name="res3_block0_conv_dw_conv2d")
    res3_block0_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res3_block0_conv_dw_conv2d,name="res3_block0_conv_dw_batchnorm")
    res3_block0_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res3_block0_conv_dw_batchnorm,name="res3_block0_conv_dw_relu")
    res3_block0_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1,pad=(0, 0), stride=(1, 1), data=res3_block0_conv_dw_relu,name="res3_block0_conv_proj_conv2d")
    res3_block0_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res3_block0_conv_proj_conv2d,name="res3_block0_conv_proj_batchnorm")

    _plus1 = mx.symbol.elemwise_add(lhs=dconv23_conv_proj_batchnorm,rhs=res3_block0_conv_proj_batchnorm, name="_plus1")

    res3_block1_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1,pad=(0, 0), stride=(1, 1), data=_plus1,name="res3_block1_conv_sep_conv2d")
    res3_block1_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res3_block1_conv_sep_conv2d,name="res3_block1_conv_sep_batchnorm")
    res3_block1_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res3_block1_conv_sep_batchnorm,name="res3_block1_conv_sep_relu")
    res3_block1_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=16, num_group=16,pad=(1, 1), stride=(1, 1), data=res3_block1_conv_sep_relu,name="res3_block1_conv_dw_conv2d")
    res3_block1_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res3_block1_conv_dw_conv2d,name="res3_block1_conv_dw_batchnorm")
    res3_block1_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res3_block1_conv_dw_batchnorm,name="res3_block1_conv_dw_relu")
    res3_block1_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=16, num_group=1,pad=(0, 0), stride=(1, 1), data=res3_block1_conv_dw_relu,name="res3_block1_conv_proj_conv2d")
    res3_block1_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res3_block1_conv_proj_conv2d,name="res3_block1_conv_proj_batchnorm")

    _plus2 = mx.symbol.elemwise_add(lhs=_plus1,rhs=res3_block1_conv_proj_batchnorm, name="_plus2")

    dconv34_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1, pad=(0, 0),stride=(1, 1), data=_plus2, name="dconv34_conv_sep_conv2d")
    dconv34_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv34_conv_sep_conv2d,name="dconv34_conv_sep_batchnorm")
    dconv34_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv34_conv_sep_batchnorm,name="dconv34_conv_sep_relu")
    dconv34_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=32, num_group=32, pad=(1, 1),stride=(2, 2), data=dconv34_conv_sep_relu,name="dconv34_conv_dw_conv2d")
    dconv34_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv34_conv_dw_conv2d,name="dconv34_conv_dw_batchnorm")
    dconv34_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv34_conv_dw_batchnorm,name="dconv34_conv_dw_relu")
    dconv34_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,pad=(0, 0), stride=(1, 1), data=dconv34_conv_dw_relu,name="dconv34_conv_proj_conv2d")
    dconv34_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv34_conv_proj_conv2d,name="dconv34_conv_proj_batchnorm")

    res4_block0_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,pad=(0, 0), stride=(1, 1), data=dconv34_conv_proj_batchnorm,name="res4_block0_conv_sep_conv2d")
    res4_block0_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res4_block0_conv_sep_conv2d,name="res4_block0_conv_sep_batchnorm")
    res4_block0_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block0_conv_sep_batchnorm,name="res4_block0_conv_sep_relu")
    res4_block0_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=32, num_group=32,pad=(1, 1), stride=(1, 1), data=res4_block0_conv_sep_relu,name="res4_block0_conv_dw_conv2d")
    res4_block0_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res4_block0_conv_dw_conv2d,name="res4_block0_conv_dw_batchnorm")
    res4_block0_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block0_conv_dw_batchnorm,name="res4_block0_conv_dw_relu")
    res4_block0_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,pad=(0, 0), stride=(1, 1), data=res4_block0_conv_dw_relu,name="res4_block0_conv_proj_conv2d")
    res4_block0_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,data=res4_block0_conv_proj_conv2d,name="res4_block0_conv_proj_batchnorm")

    _plus3 = mx.symbol.elemwise_add(lhs=dconv34_conv_proj_batchnorm,rhs=res4_block0_conv_proj_batchnorm, name="_plus3")

    res4_block1_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,
                                                        pad=(0, 0), stride=(1, 1), data=_plus3,
                                                        name="res4_block1_conv_sep_conv2d")
    res4_block1_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                         data=res4_block1_conv_sep_conv2d,
                                                         name="res4_block1_conv_sep_batchnorm")
    res4_block1_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block1_conv_sep_batchnorm,
                                                    name="res4_block1_conv_sep_relu")
    res4_block1_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=32, num_group=32,
                                                       pad=(1, 1), stride=(1, 1), data=res4_block1_conv_sep_relu,
                                                       name="res4_block1_conv_dw_conv2d")
    res4_block1_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res4_block1_conv_dw_conv2d,
                                                        name="res4_block1_conv_dw_batchnorm")
    res4_block1_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block1_conv_dw_batchnorm,
                                                   name="res4_block1_conv_dw_relu")
    res4_block1_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,
                                                         pad=(0, 0), stride=(1, 1), data=res4_block1_conv_dw_relu,
                                                         name="res4_block1_conv_proj_conv2d")
    res4_block1_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                          data=res4_block1_conv_proj_conv2d,
                                                          name="res4_block1_conv_proj_batchnorm")

    _plus4 = mx.symbol.elemwise_add(lhs=_plus3,rhs=res4_block1_conv_proj_batchnorm, name="_plus4")

    res4_block2_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,
                                                        pad=(0, 0), stride=(1, 1), data=_plus4,
                                                        name="res4_block2_conv_sep_conv2d")
    res4_block2_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                         data=res4_block2_conv_sep_conv2d,
                                                         name="res4_block2_conv_sep_batchnorm")
    res4_block2_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block2_conv_sep_batchnorm,
                                                    name="res4_block2_conv_sep_relu")
    res4_block2_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=32, num_group=32,
                                                       pad=(1, 1), stride=(1, 1), data=res4_block2_conv_sep_relu,
                                                       name="res4_block2_conv_dw_conv2d")
    res4_block2_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res4_block2_conv_dw_conv2d,
                                                        name="res4_block2_conv_dw_batchnorm")
    res4_block2_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res4_block2_conv_dw_batchnorm,
                                                   name="res4_block2_conv_dw_relu")
    res4_block2_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=32, num_group=1,
                                                         pad=(0, 0), stride=(1, 1), data=res4_block2_conv_dw_relu,
                                                         name="res4_block2_conv_proj_conv2d")
    res4_block2_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                          data=res4_block2_conv_proj_conv2d,
                                                          name="res4_block2_conv_proj_batchnorm")

    _plus5 = mx.symbol.elemwise_add(lhs=_plus4,rhs=res4_block2_conv_proj_batchnorm, name="_plus5")

    dconv45_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1, pad=(0, 0),
                                                    stride=(1, 1), data=_plus5, name="dconv45_conv_sep_conv2d")
    dconv45_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv45_conv_sep_conv2d,
                                                     name="dconv45_conv_sep_batchnorm")
    dconv45_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv45_conv_sep_batchnorm,
                                                name="dconv45_conv_sep_relu")
    dconv45_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=64, num_group=64, pad=(1, 1),
                                                   stride=(2, 2), data=dconv45_conv_sep_relu,
                                                   name="dconv45_conv_dw_conv2d")
    dconv45_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv45_conv_dw_conv2d,
                                                    name="dconv45_conv_dw_batchnorm")
    dconv45_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=dconv45_conv_dw_batchnorm,
                                               name="dconv45_conv_dw_relu")
    dconv45_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1,
                                                     pad=(0, 0), stride=(1, 1), data=dconv45_conv_dw_relu,
                                                     name="dconv45_conv_proj_conv2d")
    dconv45_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=dconv45_conv_proj_conv2d,
                                                      name="dconv45_conv_proj_batchnorm")

    res5_block0_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1,
                                                        pad=(0, 0), stride=(1, 1), data=dconv45_conv_proj_batchnorm,
                                                        name="res5_block0_conv_sep_conv2d")
    res5_block0_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                         data=res5_block0_conv_sep_conv2d,
                                                         name="res5_block0_conv_sep_batchnorm")
    res5_block0_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res5_block0_conv_sep_batchnorm,
                                                    name="res5_block0_conv_sep_relu")
    res5_block0_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=64, num_group=64,
                                                       pad=(1, 1), stride=(1, 1), data=res5_block0_conv_sep_relu,
                                                       name="res5_block0_conv_dw_conv2d")
    res5_block0_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res5_block0_conv_dw_conv2d,
                                                        name="res5_block0_conv_dw_batchnorm")
    res5_block0_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res5_block0_conv_dw_batchnorm,
                                                   name="res5_block0_conv_dw_relu")
    res5_block0_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1,
                                                         pad=(0, 0), stride=(1, 1), data=res5_block0_conv_dw_relu,
                                                         name="res5_block0_conv_proj_conv2d")
    res5_block0_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                          data=res5_block0_conv_proj_conv2d,
                                                          name="res5_block0_conv_proj_batchnorm")

    _plus6 = mx.symbol.elemwise_add(lhs=dconv45_conv_proj_batchnorm,rhs=res5_block0_conv_proj_batchnorm, name="_plus6")

    res5_block1_conv_sep_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1,
                                                        pad=(0, 0), stride=(1, 1), data=_plus6,
                                                        name="res5_block1_conv_sep_conv2d")
    res5_block1_conv_sep_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                         data=res5_block1_conv_sep_conv2d,
                                                         name="res5_block1_conv_sep_batchnorm")
    res5_block1_conv_sep_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res5_block1_conv_sep_batchnorm,
                                                    name="res5_block1_conv_sep_relu")
    res5_block1_conv_dw_conv2d = mx.symbol.Convolution(kernel=(3, 3), no_bias=True, num_filter=64, num_group=64,
                                                       pad=(1, 1), stride=(1, 1), data=res5_block1_conv_sep_relu,
                                                       name="res5_block1_conv_dw_conv2d")
    res5_block1_conv_dw_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=res5_block1_conv_dw_conv2d,
                                                        name="res5_block1_conv_dw_batchnorm")
    res5_block1_conv_dw_relu = mx.symbol.LeakyReLU(act_type="prelu", data=res5_block1_conv_dw_batchnorm,
                                                   name="res5_block1_conv_dw_relu")
    res5_block1_conv_proj_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=64, num_group=1,
                                                         pad=(0, 0), stride=(1, 1), data=res5_block1_conv_dw_relu,
                                                         name="res5_block1_conv_proj_conv2d")
    res5_block1_conv_proj_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9,
                                                          data=res5_block1_conv_proj_conv2d,
                                                          name="res5_block1_conv_proj_batchnorm")

    _plus7 = mx.symbol.elemwise_add(lhs=_plus6,rhs=res5_block1_conv_proj_batchnorm, name="_plus7")

    conv6_conv2d = mx.symbol.Convolution(kernel=(7, 7), no_bias=True, num_filter=64, num_group=1, pad=(0, 0),
                                         stride=(1, 1), data=_plus7, name="conv6_conv2d")
    conv6_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=conv6_conv2d, name="conv6_batchnorm")
    conv6_relu = mx.symbol.LeakyReLU(act_type="prelu", data=conv6_batchnorm, name="conv6_relu")
    fc1_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=128, num_group=1, pad=(0, 0),
                                       stride=(1, 1), data=conv6_relu, name="fc1_conv2d")
    fc1_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=fc1_conv2d, name="fc1_batchnorm")
    fc1_relu = mx.symbol.LeakyReLU(act_type="prelu", data=fc1_batchnorm, name="fc1_relu")
    fc2_conv2d = mx.symbol.Convolution(kernel=(1, 1), no_bias=True, num_filter=256, num_group=1, pad=(0, 0),
                                       stride=(1, 1), data=fc1_relu, name="fc2_conv2d")
    fc2_batchnorm = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=fc2_conv2d, name="fc2_batchnorm")
    fc2_relu = mx.symbol.LeakyReLU(act_type="prelu", data=fc2_batchnorm, name="fc2_relu")

    conv6_3 = mx.symbol.FullyConnected(num_hidden=212, data=fc2_relu, name="conv6_3")
    bn6_3 = mx.symbol.BatchNorm(fix_gamma=False, momentum=0.9, data=conv6_3, name="bn6_3")

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:

        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                         grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target,
                               op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])

    return group