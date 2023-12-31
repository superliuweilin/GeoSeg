from torch import nn
import torch
from torch.nn.modules import conv
from torch.nn.modules.conv import Conv2d
from einops import rearrange
from torch.nn import init

class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model,1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input) #(bs,nq,1)
        weight_i = torch.softmax(i, dim=1) #bs,nq,1
        context_score = weight_i * self.fc_k(input) #bs,nq,d_model
        context_vector = torch.sum(context_score,dim=1,keepdim=True) #bs,1,d_model
        v = self.fc_v(input) * context_vector #bs,nq,d_model
        out = self.fc_o(v) #bs,nq,d_model

        return out


def conv_bn(inp,oup,kernel_size=3,stride=1):
    return nn.Sequential(
        nn.Conv2d(inp,oup,kernel_size=kernel_size,stride=stride,padding=kernel_size//2),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.ln(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim,dropout) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads,head_dim,dropout):
        super().__init__()
        inner_dim=heads*head_dim
        project_out=not(heads==1 and head_dim==dim)
        '''如果head==1并且head_dim==dim,project_out=False,否则project_out=True'''

        self.heads=heads
        self.scale=head_dim**-0.5

        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        # print(x.shape)
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        # print('qkv', qkv[0].shape)
        #chunk对数组进行分组,3表示组数，dim表示在哪一个维度.这里返回的qkv为一个包含三组张量的元组
        q,k,v=map(lambda t:rearrange(t,'b p n (h d) -> b p h n d',h=self.heads),qkv)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        out=torch.matmul(attn,v)
        out=rearrange(out,'b p h n d -> b p n (h d)')
        # print('self.to_out', self.to_out(out).shape)
        return self.to_out(out)





class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,head_dim,mlp_dim,dropout=0.):
        super().__init__()
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,MobileViTv2Attention(dim)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))


    def forward(self,x):
        out=x
        for att,ffn in self.layers:
            out=out+att(out)
            out=out+ffn(out)
        return out

class MobileViTAttention(nn.Module):
    def __init__(self,in_channel=3,dim=512,kernel_size=3,patch_size=7,depth=3,mlp_dim=1024):
        super().__init__()
        self.ph,self.pw=patch_size,patch_size
        self.conv1=nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)
        self.conv2=nn.Conv2d(in_channel,dim,kernel_size=1)

        self.trans=Transformer(dim=dim,depth=depth,heads=8,head_dim=64,mlp_dim=mlp_dim)

        self.conv3=nn.Conv2d(dim,in_channel,kernel_size=1)
        self.conv4=nn.Conv2d(2*in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)

    def forward(self,x):
        # y=x.clone() #bs,c,h,w
        y = x

        ## Local Representation
        y=self.conv2(self.conv1(x)) #bs,dim,h,w

        ## Global Representation
        _,_,h,w=y.shape
        # print(y.shape)
        y=rearrange(y,'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',ph=self.ph,pw=self.pw) #bs,h,w,dim
        # print('y_a',y.shape)
        y=self.trans(y)
        # print('y_at', y.shape)
        y=rearrange(y,'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',ph=self.ph,pw=self.pw,nh=h//self.ph,nw=w//self.pw) #bs,dim,h,w

        ## Fusion
        y=self.conv3(y) #bs,dim,h,w
        # y=torch.cat([x,y],1) #bs,2*dim,h,w
        # y=self.conv4(y) #bs,c,h,w

        return y


class MV2Block(nn.Module):
    def __init__(self,inp,out,stride=1,expansion=1):
        super().__init__()
        self.stride=stride
        hidden_dim=inp*expansion
        self.use_res_connection=stride==1 and inp==out

        if expansion==1:
            self.conv=nn.Sequential(
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=self.stride,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim,out,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(out)
            )
        else:
            self.conv=nn.Sequential(
                nn.Conv2d(inp,hidden_dim,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=1,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim,out,kernel_size=1,stride=1,bias=False),
                nn.SiLU(),
                nn.BatchNorm2d(out)
            )
    def forward(self,x):
        if(self.use_res_connection):
            out=x+self.conv(x)
        else:
            out=self.conv(x)
        return out

class MobileViT(nn.Module):
    def __init__(self,image_size,dims,channels,num_classes,depths=[2,4,3],expansion=4,kernel_size=3,patch_size=2):
        super().__init__()
        ih,iw=image_size,image_size
        ph,pw=patch_size,patch_size
        assert iw%pw==0 and ih%ph==0

        self.conv1=conv_bn(3,channels[0],kernel_size=3,stride=patch_size)
        self.mv2=nn.ModuleList([])
        self.m_vits=nn.ModuleList([])


        self.mv2.append(MV2Block(channels[0],channels[1],1))
        self.mv2.append(MV2Block(channels[1],channels[2],2))
        self.mv2.append(MV2Block(channels[2],channels[3],1))
        self.mv2.append(MV2Block(channels[2],channels[3],1)) # x2
        self.mv2.append(MV2Block(channels[3],channels[4],2))
        self.m_vits.append(MobileViTAttention(channels[4],dim=dims[0],kernel_size=kernel_size,patch_size=patch_size,depth=depths[0],mlp_dim=int(2*dims[0])))
        self.mv2.append(MV2Block(channels[4],channels[5],2))
        self.m_vits.append(MobileViTAttention(channels[5],dim=dims[1],kernel_size=kernel_size,patch_size=patch_size,depth=depths[1],mlp_dim=int(4*dims[1])))
        self.mv2.append(MV2Block(channels[5],channels[6],2))
        self.m_vits.append(MobileViTAttention(channels[6],dim=dims[2],kernel_size=kernel_size,patch_size=patch_size,depth=depths[2],mlp_dim=int(4*dims[2])))

        
        self.conv2=conv_bn(channels[-2],channels[-1],kernel_size=1)
        # self.pool=nn.AvgPool2d(image_size//32,1)
        # self.fc=nn.Linear(channels[-1],num_classes,bias=False)

    def forward(self,x):
        # print(x.shape)
        y=self.conv1(x) #
        y=self.mv2[0](y)
        y=self.mv2[1](y) #
       
        y=self.mv2[2](y)
        y=self.mv2[3](y)
        res1 = y
        # print(y.shape)
        y=self.mv2[4](y) #
        # print('y=', y.shape)
        y=self.m_vits[0](y)
        # print('y_vit', y.shape)
        res2 = y
        y=self.mv2[5](y) #
        
        y=self.m_vits[1](y)
        # print(y.shape)
        res3 = y

        y=self.mv2[6](y) #
        y=self.m_vits[2](y)

        y=self.conv2(y)
        # print(y.shape)
        res4 = y
        # y=self.pool(y).view(y.shape[0],-1) 
        # # print(y.shape)
        
        # y=self.fc(y)
        return res1, res2, res3, res4

def mobilevit_xxs():
    dims=[60,80,96]
    channels= [16, 16, 24, 24, 48, 64, 80, 320]
    return MobileViT(224,dims,channels,num_classes=1000)

def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    return MobileViT(224, dims, channels, num_classes=1000)

def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    return MobileViT(768, dims, channels, num_classes=6)


def count_paratermeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    input=torch.randn(1, 3, 768, 768)
    

    ### mobilevit_xxs
    mvit_xxs=mobilevit_s()
    outs = mvit_xxs(input)
    for out in outs:
        print(out.shape)
    
    

    # # ### mobilevit_xs
    # # mvit_xs=mobilevit_xs()
    # # out=mvit_xs(input)
    # # # print(out.shape)


    # ### mobilevit_s
    # mvit_s=mobilevit_s()
    # print(count_paratermeters(mvit_s))
    # out=mvit_s(input)
    # # print(out.shape)
    # attention = Attention(dim=128, heads=8, head_dim=64, dropout=0.2)
    # out = attention(input)
    # print(out.shape)

    