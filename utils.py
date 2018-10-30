import math
import torch as th
import numpy as np
from functools import reduce

def formatInput2Tuple(input,typeB,numel,strict = True):
    assert(isinstance(input,(tuple,typeB))),"input is expected to be of type " \
        "tuple or of type " + str(typeB)[8:-2] + " but instead an input of "\
        +"type "+str(type(input))+" was provided."
    
    if isinstance(input,typeB):
        input = (input,)*numel
    
    if strict :
        assert(len(input) == numel), "An input of size "+str(numel)+" is expected "\
            "but instead input = "+str(input)+ " was provided."
    else:
        if len(input) < numel:
            input = input + (input[-1],)*(numel-len(input))
        elif len(input) > numel:
            input = input[0:numel]
        
    return tuple(typeB(i) for i in input)

def reverse(input,dim=0) :
    r"""Reverses the specified dimension of the input tensor."""
    Dims = input.dim()
    assert (dim < Dims), "The selected dimension (arg 2) exceeds the tensor's dimensions."
    idx = th.arange(input.size(dim)-1,-1,-1).type_as(input).long()
    return input.index_select(dim,idx)

def periodicPad2D(input,pad = 0):
    r"""Pads circularly the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.empty(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = out[...,out.size(-2)-pad[1]-pad[0]:out.size(-2)-pad[1],:]
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = out[...,pad[0]:pad[0]+pad[1],:]
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = out[...,:,out.size(-1)-pad[3]-pad[2]:out.size(-1)-pad[3]]
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = out[...,:,pad[2]:pad[2]+pad[3]]    
    
    if sflag:
        out.squeeze_()
        
    return out

def periodicPad_transpose2D(input,crop = 0):
    r"""Adjoint of the periodicPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:crop[0]+crop[1],:] += out[...,-crop[1]::,:]
    
    # Bottom 
    if crop[1] != 0:
        out[...,-crop[0]-crop[1]:-crop[1],:] += out[...,0:crop[0],:]
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:crop[2]+crop[3]] += out[...,-crop[3]::]
    
    # Right
    if crop[3] != 0:
        out[...,-crop[2]-crop[3]:-crop[3]] += out[...,0:crop[2]]
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    if sflag:
        out.squeeze_()
        
    return out


def zeroPad2D(input,pad = 0):
    r"""Pads with zeros the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""

    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
        
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."    
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.zeros(sz).type_as(input)
    out[...,pad[0]:sz[-2]-pad[1]:1,pad[2]:sz[-1]-pad[3]:1] = input
    
    if sflag:
        out.squeeze_()
    
    return out

def crop2D(input,crop):
    r"""Cropping the spatial dimensions (last two dimensions) of the 
    input tensor. This is the adjoint operation of zeroPad2D. Crop specifies 
    the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT]. If crop is an integer 
    then each direction is cropped by the same amount. In order to achieve a 
    different amount of cropping in each direction of the  tensor, crop needs 
    to be a tuple."""    
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())    
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input[...,crop[0]:sz[-2]-crop[1]:1,crop[2]:sz[-1]-crop[3]:1]
    
    if sflag:
        out.unsqueeze_()
    
    return out
    
def symmetricPad2D(input,pad = 0):
    r"""Pads symmetrically the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple."""
          
    # pad = [top,bottom,left,right]
    
    if isinstance(pad,int):
        assert(pad >= 0), """Pad must be either a non-negative integer 
        or a tuple."""
        pad = (pad,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)
             
    assert(isinstance(pad,tuple) and len(pad) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (pad[0] >= 0 and pad[1] >= 0 and pad[2] >= 0 and pad[3] >= 0), \
            "Padding must be non-negative in each dimension."
            
    assert(pad[0] <= sz[-2] and pad[1] <= sz[-2] and \
           pad[2] <= sz[-1] and pad[3] <= sz[-1]), \
    "The padding values exceed the tensor's dimensions."
    
    sz[-1] = sz[-1] + sum(pad[2::])
    sz[-2] = sz[-2] + sum(pad[0:2])
    
    out = th.zeros(sz).type_as(input)
    
    # Copy the original tensor to the central part
    out[...,pad[0]:out.size(-2)-pad[1], \
        pad[2]:out.size(-1)-pad[3]] = input
    
    # Pad Top
    if pad[0] != 0:
        out[...,0:pad[0],:] = reverse(out[...,pad[0]:2*pad[0],:],-2)
    
    # Pad Bottom
    if pad[1] != 0:
        out[...,out.size(-2)-pad[1]::,:] = reverse(out[...,out.size(-2)
            -2*pad[1]:out.size(-2)-pad[1],:],-2)
    
    # Pad Left
    if pad[2] != 0:
        out[...,:,0:pad[2]] = reverse(out[...,:,pad[2]:2*pad[2]],-1)
    
    # Pad Right
    if pad[3] != 0:
        out[...,:,out.size(-1)-pad[3]::] = reverse(out[...,:,out.size(-1)
            -2*pad[3]:out.size(-1)-pad[3]],-1)    
    
    if sflag:
        out.squeeze_()
        
    return out


def symmetricPad_transpose2D(input,crop = 0):
    r"""Adjoint of the SymmetricPad2D operation which amounts to a special type
    of cropping. CROP specifies the amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
    If crop is an integer then each direction is cropped by the same amount. In
    order to achieve a different amount of cropping in each direction of the 
    tensor, crop needs to be a tuple."""          
    
    # crop = [top,bottom,left,right]
    
    if isinstance(crop,int):
        assert(crop >= 0), """Crop must be either a non-negative integer 
        or a tuple."""
        crop = (crop,)*4  
        
    sflag = False
    if input.dim() == 1:
        sflag = True
        input = input.unsqueeze(1)        
             
    assert(isinstance(crop,tuple) and len(crop) == 4), \
    " A tuple with 4 values for padding is expected as input."
        
    sz = list(input.size())
    
    assert (crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0), \
            "Crop must be non-negative in each dimension."    
    
    assert (crop[0] + crop[1] <= sz[-2] and crop[2] + crop[3] <= sz[-1]), \
            "Crop does not have valid values."
    
    out = input.clone()
    
    # Top
    if crop[0] != 0:
        out[...,crop[0]:2*crop[0],:] += reverse(out[...,0:crop[0],:],-2)
    
    # Bottom 
    if crop[1] != 0:
    # out[...,sz[-2]-2*crop[1]:sz[-2]-crop[1],:] += reverse(out[...,sz[-2]-crop[1]::,:],-2) 
        out[...,-2*crop[1]:-crop[1],:] += reverse(out[...,-crop[1]::,:],-2) 
    
    # Left 
    if crop[2] != 0:
        out[...,crop[2]:2*crop[2]] += reverse(out[...,0:crop[2]],-1)
    
    # Right
    if crop[3] != 0:
    # out[...,sz[-1]-2*crop[3]:sz[-1]-crop[3],:] += reverse(out[...,sz[-1]-crop[3]::,:],-1) 
        out[...,-2*crop[3]:-crop[3]] += reverse(out[...,-crop[3]::],-1) 
    
    if crop[1] == 0:
        end_h = sz[-2]+1 
    else:
        end_h = sz[-2]-crop[1]
        
    if crop[3] == 0:
        end_w = sz[-1]+1
    else:
        end_w = sz[-1]-crop[3]
        
    out = out[...,crop[0]:end_h,crop[2]:end_w]
    
    
    if sflag:
        out.squeeze_()
        
    return out

def pad2D(input,pad=0,padType='zero'):
    r"""Pads the spatial dimensions (last two dimensions) of the 
    input tensor. PAD specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero","symmetric" and "periodic". """
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return zeroPad2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad2D(input,pad)
    elif padType == 'periodic':
        return periodicPad2D(input,pad)
    else:
        raise NotImplementedError("Unknown padding type.")

def pad_transpose2D(input,pad=0,padType='zero'):
    r"""Transpose operation of pad2D. PAD specifies the amount of padding as 
    [TOP, BOTTOM, LEFT, RIGHT].
    If pad is an integer then each direction is padded by the same amount. In
    order to achieve a different amount of padding in each direction of the 
    tensor, pad needs to be a tuple. PadType specifies the type of padding.
    Valid padding types are "zero" and "symmetric". """    
    
    pad = formatInput2Tuple(pad,int,4)
    
    if sum(pad) == 0:
        return input
    
    if padType == 'zero':
        return crop2D(input,pad)
    elif padType == 'symmetric':
        return symmetricPad_transpose2D(input,pad)
    elif padType == 'periodic':
        return periodicPad_transpose2D(input,pad)
    else:
        raise NotImplementedError("Uknown padding type.")


def shift(x,s,bc='circular'):
    """ Shift operator that can treat different boundary conditions. It applies 
    to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero'| 'inf')
        (Default: 'circular')
    """
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
        s = s + (0,) * (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")

    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered         
    indices = [slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = (np.arange(0,m)-s[i])%m
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                idx = indices[:]
                if s[i] > 0: # right shift                    
                    idx[i] = list(range(s[i]-1,-1,-1)) + list(range(0,x.shape[i]-s[i]))
                else: # left shift
                    idx[i] = list(range(-s[i],x.shape[i])) + \
                    list(range(x.shape[i]-1,x.shape[i]+s[i]-1,-1))
                
                xs = xs[tuple(idx)]
    elif bc == 'zero':
        xs=th.zeros_like(x)        
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    elif bc == 'inf':
        xs=np.inf*th.ones_like(x)        
        idx_x=indices[:]
        idx_xs=indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:       
                if s[i] > 0: # right shift
                    idx_x[i] = slice(0,x.shape[i]-s[i])
                    idx_xs[i] = slice(s[i],x.shape[i])
                else: # left shift
                    idx_x[i] = slice(-s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]+s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def shift_transpose(x,s,bc='circular'):
        
    r""" Transpose of the shift operator that can treat different boundary conditions. 
    It applies to a tensor of arbitrary dimensions. 
    ----------
    Usage: xs = shift_transpose(x,(0,1,-3,3),'reflexive')
    ----------
    Parameters
    ----------
    x : tensor.
    s : tuple that matches the dimensions of x, with the corresponding shifts.
    bc: String with the prefered boundary conditions (bc='circular'|'reflexive'|'zero'| 'inf')
        (Default: 'circular')
    """   
    
    if not isinstance(bc, str):
        raise Exception("bc must be of type string")
       
    if not reduce(lambda x,y : x and y, [isinstance(k,int) for k in s]):
        raise Exception("s must be a tuple of ints")
           
    if len(s) < x.dim():
       s = s + (0,)* (x.dim()-len(s))        
    elif len(s) > x.dim():
        print("The shift values will be truncated to match the " \
        +"dimensions of the input tensor. The trailing extra elements will" \
        +" be discarded.")
        s = s[0:x.dim()]
    
    if reduce(lambda x,y : x or y, [ math.fabs(s[i]) > x.shape[i] for i in range(x.dim())]):
        raise Exception("The shift steps should not exceed in absolute values"\
        +" the size of the corresponding dimensions.")
        
    # use a list sequence instead of a tuple since the latter is an 
    # immutable sequence and cannot be altered 
    indices=[slice(0,x.shape[0])]
    for i in range(1,x.dim()):
        indices.append(slice(0,x.shape[i]))
        
    if bc == 'circular':
        xs = x.clone() # make a copy of x
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                m = x.shape[i]
                idx = indices[:]                
                idx[i] = (np.arange(0,m)+s[i])%m
                xs = xs[tuple(idx)]
    elif bc == 'reflexive':
        y=x.clone()
        for i in range(x.dim()):
            xs = th.zeros_like(x)
            idx_x_a = indices[:]
            #idx_x_b = indices[:]
            idx_xs_a = indices[:]
            idx_xs_b = indices[:]
            if s[i] == 0:
                xs = y.clone()
            else:
                if s[i] > 0:
                    idx_xs_a[i] = slice(0,-s[i])
                    idx_xs_b[i] = slice(0,s[i])
                    idx_x_a[i] = slice(s[i],None)
                    #idx_x_b[i] = slice(s[i]-1,None,-1) #Pytorch does not 
                    # support negative steps
                else:
                    idx_xs_a[i] = slice(-s[i],None)
                    idx_xs_b[i] = slice(s[i],None)
                    idx_x_a[i] = slice(0,s[i])
                    #idx_x_b[i] = slice(-1,s[i]-1,-1) #Pytorch does not 
                    # support negative steps
                
                xs[tuple(idx_xs_a)] = y[tuple(idx_x_a)]
                xs[tuple(idx_xs_b)] += reverse(y[tuple(idx_xs_b)],dim = i)
                #xs[tuple(idx_xs_b)] += y[tuple(idx_x_b)]
                y = xs.clone()
        
    elif bc == 'zero':
        xs = th.zeros_like(x)        
        idx_x = indices[:]
        idx_xs = indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                if s[i] < 0: 
                    idx_x[i] = slice(0,x.shape[i]+s[i])
                    idx_xs[i] = slice(-s[i],x.shape[i])
                else: 
                    idx_x[i] = slice(s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]-s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
        
    elif bc == 'inf':
        xs = np.inf*th.ones_like(x)        
        idx_x = indices[:]
        idx_xs = indices[:]
        for i in range(x.dim()):
            if s[i] == 0:
                continue
            else:
                if s[i] < 0: 
                    idx_x[i] = slice(0,x.shape[i]+s[i])
                    idx_xs[i] = slice(-s[i],x.shape[i])
                else: 
                    idx_x[i] = slice(s[i],x.shape[i])
                    idx_xs[i] = slice(0,x.shape[i]-s[i])
        
        xs[tuple(idx_xs)] = x[tuple(idx_x)]
        
    else:
        raise Exception("Unknown boundary conditions")
    
    return xs

def im2patch(input,patchSize,stride=1) :
    r""" im2patch extracts all the valid patches from the input which is a 3D 
    or 4D tensor of size B x C x H x W. The extracted patches are of size 
    patchSize and they are extracted with an overlap equal to stride. The 
    output is of size B x C*P x PH x PW where P is the total number of elements
    in the patch, while PH and PW is the number of patches in the horizontal and
    vertical axes, respectively.
    """
    assert(input.dim() >= 3 and input.dim() < 5), "A 3D or 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    
    if input.dim() == 3:
        input = input.unsqueeze(0)
        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    batch, Nc = input.shape[0:2] 
    
    if Nc != 1:
        input = input.view(batch*Nc,1,input.shape[2],input.shape[3])
    
    P = th.conv2d(input,h,stride = stride)
    
    if Nc != 1:
        P = P.view(batch,Nc*Pn,P.shape[2],P.shape[3])
    
    return P

def patch2im(input,shape,patchSize,stride=1) :
    r""" patch2im is the transpose operation of im2patch.
    
    shape : is the size of the original tensor from which the patches where 
    extracted.
    """
    assert(input.dim() == 4), "A 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]

        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    batch = shape[0]
    Nc = math.floor(input.shape[1]/Pn);
    if Nc != 1:
        input = input.view(batch*Nc,input.shape[1]/Nc,input.shape[2],input.shape[3])
    
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    out = th.conv_transpose2d(input,h,stride = stride)
        
    if Nc != 1:
        out = out.view(batch,Nc*out.shape[1],out.shape[2],out.shape[3])
    
    if reduce(lambda x,y : x or y,[out.shape[i] < shape[i] for i in range(4)]):
        out = th.nn.functional.pad(out,(0,shape[3]-out.shape[3],0,shape[2]-out.shape[2]))
    
    return out

def im2patch_sinv(input,shape,patchSize,stride=1) :
    r""" im2patch_sinv is the pseudo inverse of im2patch.
    
    shape : is the size of the original tensor from which the patches where 
    extracted.
    """
    assert(input.dim() == 4), "A 4D tensor is expected."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    
    if len(patchSize) < 2:
        patchSize  *=  2
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]

        
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    batch = shape[0]
    Nc = math.floor(input.shape[1]/Pn);
    if Nc != 1:
        input = input.view(batch*Nc,input.shape[1]/Nc,input.shape[2],input.shape[3])
    
    h = th.eye(Pn).type(input.type())
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    out = th.conv_transpose2d(input,h,stride = stride)
        
    if Nc != 1:
        out = out.view(batch,Nc*out.shape[1],out.shape[2],out.shape[3])
    
    D = compute_patch_overlap(shape,patchSize,stride)
    D = D.type(input.type())
    out = out.div(D)
    
    if reduce(lambda x,y : x or y,[out.shape[i] < shape[i] for i in range(4)]):
        out = th.nn.functional.pad(out,(0,shape[3]-out.shape[3],0,shape[2]-out.shape[2]))
    
    return out


def compute_patch_overlap(shape,patchSize,stride=1,padding=0,GPU=False,dtype = 'f'):
    r""" Returns a tensor whose dimensions are equal to 'shape' and it 
    indicates how many patches extracted from the image (the patches are of 
    size patchSize and are extracted using a specified stride) each pixel of 
    the image contributes. 

    For example below is the array which indicates how many times each pixel
    at the particular location of an image of size 16 x 16 has been found in 
    any of the 49 4x4 patches that have been extracted using a stride=2.

    T = 
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
     1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1


     Based on this table the pixel at the location (3,2) has been used in 4
     different patches while the pixel at the location (15,4) has been used in
     2 different patches."""
     
    assert(isinstance(shape,tuple)), "shape is expected to be a tuple."
    assert(isinstance(patchSize,tuple)), "patchSize is expected to be a tuple."
    if len(shape) < 4:
        shape = (1,)*(4-len(shape)) + shape
    elif len(shape) > 4:
        shape = shape[0:3]
    
    if len(patchSize) < 2:
        patchSize = patchSize * 2
    
    if dtype == 'f' : 
        dtype = th.FloatTensor
    elif dtype == 'd' :
        dtype = th.DoubleTensor
    else:
        raise Exception("Supported data types are 'f' (float) and 'd' (double).")
    
    
    shape_ = (shape[0]*shape[1],1,shape[2],shape[3])
    
    
    Pn = reduce(lambda x,y : x*y, patchSize[0:2])
    h = th.eye(Pn).type(dtype)
    h = h.view(Pn,1,patchSize[0],patchSize[1])
    
    x = th.ones(shape_).type(dtype)
    
    if th.cuda.is_available() and GPU:
        x = x.cuda()
        h = h.cuda()
    
    T = th.conv2d(x,h,stride = stride, padding = padding)
    T = th.conv_transpose2d(T,h,stride = stride, padding = padding)
    
    return T.view(shape)



def knnpatch(f, blocksize, searchwin, stride = 1, K=10, GPU=False):
# [KNN,KNN_D]=KNNPATCH(f,blocksize,searchwin) returns a KNN-field 
#(K-Nearest Neighbors) for each patch of size Hp x Wp in a search window of
#size (2*Hw+1) x (2*Ww+1).
#
# ========================== INPUT PARAMETERS (required) ==================
# Parameters    Values description
# =========================================================================
# f             Vector/Scalar-valued image of size Nx x Ny x Nc x B, 
#               where Nc: number of channels and B: number of images.
# blocksize     Vector [Hp Wp] where Hp and Wp is the height and width of  
#               the patches extracted from the image f.
# searchwin     Vector [Hw Ww] where 2*Hw+1 and 2*Ww+1 is the height and 
#               width of the search window.
# ======================== OPTIONAL INPUT PARAMETERS ======================
# Parameters    Values' description
# stride        [Sx Sy] where Sx and Sy are the step-sizes in x- and
#               y-directions. (Default: stride=[1 1]).
# padSize       specifies the amount of padding of the image as 
#               [TOP, BOTTOM, LEFT, RIGHT]. 
# padType       'zero' || 'symmetric' indicates the type of padding.
#               (Default: padType='zero').
# W             Weights used in the distance measure between two patches:
#                       Hp/2   Wp/2
#               d(a,b)= S      S  W(i,j)|f(a_x+i,a_y+j)-f(b_x+i,b_y+j)|^2
#                      i=-Hp/2 j=-Wp/2
#               (Default: nweights=ones(blocksize)). Note that W
#               must be a symmetric matrix. 
# K             Number of closest neighbors (Default: 10). 
# sorted        If sorted is set to true then knn is sorted based on the
#               patch-distance (from smaller to larger).
# patchDist    'euclidean' || 'abs' indicates the type of the 
#               patch distance (Default:'euclidean').
# transform     If set to true then the patch similarity takes place in the
#               gradient domain. (Default : false)
# GPU           True || False. Flag which indicates if the function 
#               will use the GPU support or not. (Default: False)
# =========================================================================
# ========================== OUTPUT PARAMETERS ============================
# knn           KNN field of size Np x K x B which contains the 
#               coordinates of the center of the K closest patches
#               (including the patch itself). 
# knn_D         KNN field of size Np x K x B which contains the distances
#               of the K closest patches (including the patch itself).
# LUT           Look-up table which maps the image coordinates of the patch
#               centers to the index of the patches extracted by im2patch.
# =========================================================================

# if K = 1 then we do not need to make any search.

    padSize = (0,0,0,0)
    padType = 'zero'
    W = th.ones(blocksize).view(1,1, blocksize[0], blocksize[1])
    Ni, Nc, Nx, Ny = f.size()
    pH = int((Nx-blocksize[0])/stride+1)
    pW = int((Ny-blocksize[1])/stride+1)
    patchDims = (pH, pW)
    #print(patchDims)
    patch_num = patchDims[0]*patchDims[1]
    K = K-1
    knn = th.zeros(Ni, patchDims[0], patchDims[1], K).int()
    knn_D = th.zeros(Ni, patchDims[0], patchDims[1], K)
    if GPU:
        knn = knn.cuda()
        knn_D = knn_D.cuda()
    w = blocksize
    wc = (math.floor((w[0]+1)/2)-1, math.floor((w[1]+1)/2)-1)
    #print('Patch center:', wc)
    T = th.arange(Nx*Ny).view(Nx, Ny).int()
    T = T[wc[0]:-w[0]+wc[0]+1:stride, wc[1]:-w[1]+wc[1]+1:stride]
    
   
    if K == 0:
        if GPU:
            T = T.cuda()
        knn = th.stack([T]*Ni)
        LUT = []
        return knn, LUT
    
    sx = th.arange(-searchwin[0], searchwin[0]+1, stride).int()
    sy = th.arange(-searchwin[0], searchwin[0]+1, stride).int()
    
    ctr = 0
    for kx in sx:
        for ky in sy:
            if ~(kx==0 and ky==0):# We do not check the (0,0)-offset case since in
      #this case each patch is compared to itself and the distance would be
      #zero. We add this weight at the end. (This is why we redefine above
      #K as K=K-1.)
                #print(int(kx),int(ky))
                #print(ctr)
                E = (f - shift(f, (0,0,-int(kx), -int(ky)), bc = 'inf'))**2
                D = th.conv2d(E, W, stride=1)
                D = th.squeeze(D)
                if ctr < K:
                    knn[:, :, :, ctr] = th.stack([T+kx*Ny+ky]*Ni)
                    knn_D[:,:,:, ctr] = D
            
                else:
                    # Check if the maximum element of knn_D is greater than the value of D
                    # If yes, we keep the minimum weight. At the end we will have kept
                    # the K smallest weights. 
                    M, idx = th.max(knn_D, dim = 3)
                    idx = th.squeeze(idx)
                    idx_ = th.arange(Ni*patch_num)*K + idx.view(-1)
                    
                    M = th.squeeze(M)
                    mask = (M > D)
                    M = th.min(M, D)
                    knn_D.view(-1)[idx_] = M.view(-1)
                    knn_D.view(Ni, patchDims[0], patchDims[1], K)
                    
                    R = th.stack([T+kx*Ny+ky]*Ni)
                    idx_ = idx_[mask.view(-1)==1]
                   
                    knn.view(-1)[idx_] = R.view(-1)[mask.view(-1)==1]
                ctr += 1
    V = th.stack([T]*Ni).view(Ni, patchDims[0], patchDims[1], 1)
    knn = th.cat([V, knn], dim = 3)
    V = th.zeros(Ni, patchDims[0], patchDims[1], 1)
    knn_D = th.cat([V, knn_D], dim = 3)
    LUT = th.zeros(Nx*Ny, 1).int()
    for k in range(patch_num):
        LUT[T.contiguous().view(-1)[k]] = k
    return knn, knn_D, LUT
    
    
    
def blockmatch(input, blocksize, searchwin, stride = 1, Nbrs= 10, GPU = False):
    

    patched_input = im2patch(input, blocksize, stride )
    patched_input = patched_input.transpose(1,2).transpose(2, 3)
    
    Ni, pH, pW, K = patched_input.size()
    patch_num = pH*pW
    
    knn_idx, knn_D, LUT = knnpatch(input, blocksize, searchwin, stride = 1, K=Nbrs, GPU=False)
    
   
    ### save the output of knnpatch
    ####
    
    idx_ = LUT[knn_idx.view(-1).long()]
    idx_ = idx_.view(Ni, patch_num, Nbrs)
    
    ### transform patch indeces to global ones
    
    for i in range(Ni):
        idx_[i] +=i*patch_num
    
    return patched_input.contiguous().view(Ni*patch_num, K)[idx_.view(Ni*patch_num,Nbrs).long()].view(Ni,\
                                                                                            patch_num,Nbrs,K),\
                                                                                            patched_input,\
                                                                                            knn_idx, idx_, knn_D, LUT
        
def iblockmatch(input, idx, GPU = False):
    # idx - global indexes
    Ni, num_patch, Nbrs, K = input.size()
    iD = th.FloatTensor(Ni*num_patch, K)
    if GPU:
        iD = iD.cuda()
    #mask = [idx.view(-1)==i for i in range(Ni*num_patch)]
    #mask = th.stack(mask)
    #n = mask.sum(dim=1).float()
  
    n = th.histc(idx.view(-1).float(), bins=Ni*num_patch, min=-1, max=Ni*num_patch)
   
    #print(n.size())
    for i in range(Ni*num_patch):
        iD[i] = input.view(Ni*num_patch*Nbrs, K)[(idx.view(-1)==i).nonzero()].sum(dim=0)/n[i]
        
    return iD
        
        
       