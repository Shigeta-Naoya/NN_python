# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:24:51 2020
基本的な関数とレイヤーごとに順方向，逆方向の伝搬を行うクラス
@author: magic
"""
import numpy as np

#ソフトマックス関数
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

                              
#恒等関数
def identity_function(x):
    return x


#交差エントロピー
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
     # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#relu関数のクラス
# =============================================================================
#x:入力
# out:自身の出力
#dout:逆伝搬の入力
# dx:逆伝搬の出力
# mask:自身の出力が負の場合Trueになる配列
# =============================================================================

class Relu:
    #初期化　インスタンス変数mask
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        #0以下の場合にマスクをかけておく
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx


#sigmoid関数のクラス
# =============================================================================
#x:入力
#out:自身の出力
#dout:逆伝搬の入力
#dx:逆伝搬の出力
# =============================================================================
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        
        return dx
        
#affineのクラス(重み計算関係)
# =============================================================================
#W:重み
#b:バイアス
#x:入力
# out:自身の出力
#dx:逆伝搬の入力
#dout:逆伝搬の出力
# =============================================================================
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
    
#Softmax-with-loss
# =============================================================================
#x:入力
#t:教師データ
#dx:逆伝搬の出力
# =============================================================================
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    #損失
        self.y = None       #softmaxの出力
        self.t = None       #教師データ(onehot)
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx