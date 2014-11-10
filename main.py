#coding: utf-8

DEBUG = True
#DEBUG = False

from decimal import *

import sys
import numpy as np
from hierarchical_model import BinTree
from network import LinearNetwork

def run():
    SEED = 29431101
    n_epochs = 1000
    n3 = 10000  # Tree数
    depth = 3
    n2 = 3  # 中間層数
    n1 = 2**(depth - 1)
    p = 0.1  # 反転率
    mu = 0.0005
    a0 = 0.00002

    svd = svd_reduce  # 使用するsvd

    theory_filename = "theory_result.dat"
    experiment_filename = "experiment_result.dat"
    err_filename = "err.dat"

    rng = np.random.RandomState(SEED)

    # 教師信号の作成
    if DEBUG: print "create inst[]"
    inst = create_instruction_signals(n_in = n1, n_out = n3, depth = depth, p = p, rng = rng)  # inst[0]: 入力,  inst[1]: 出力
    if DEBUG: print "inst[0].shape:", inst[0].shape
    if DEBUG: print "inst[1].shape:", inst[1].shape

    # ネットワークの作成
    if DEBUG: print "create network"
    network = LinearNetwork(l = 3, n = (n1, n2, n3), rng = rng)
    #network = LinearNetwork(l = 3, n = (n1, n2, n3))
    if DEBUG: print "W0.shape:", network.W[0].shape
    if DEBUG: print "W1.shape:", network.W[1].shape

    s31 = np.dot(inst[1], inst[0].T)  # Sigma31
    if DEBUG: print "S31 (shape:", s31.shape, ")\n", s31
    s11 = np.dot(inst[0], inst[0].T)  # Sigma11
    if DEBUG: print "S31 (shape:", s11.shape, ")\n", s11

    # do SVD
    (U, S, s, V) = svd(s31)  # s31 = U, S, V.T
    if DEBUG: print "U.shape", U.shape
    if DEBUG: print "S.shape", S.shape
    if DEBUG: print "V.shape", V.shape
    if DEBUG: print "UUT", np.dot(U, U.T)
    if DEBUG: print "VVT", np.dot(V, V.T)
    if DEBUG: print "s31 ?=", np.dot(U, np.dot(S, V))

    # 更新量を計算 -> 更新
    W0bar = np.dot(network.W[0], V)  # 本当は(W0, inv(V.T))だが，V.T = inv(V) とした．
    #if DEBUG: print "W0bar (shape:", W0bar.shape, ")\n", W0bar
    W1bar = np.dot(U.T, network.W[1])  # 本当は(inv(U), W1)だが，U.T = inv(U) とした．
    #if DEBUG: print "W1bar (shape:", W0bar.shape, ")\n", W1bar
    theory_strength = []
    experiment_strength = []
    err = []

    # 理論のstrength(t = 0)
    for i in xrange(len(s)):
        theory_strength.append(
            [calculate_strength(t = 0, s = s[i], a0 = a0, tau = 1.0/mu)]
        )

    # 実際のstrength(t = 0)
    netout = network.run(inst[0])
    netout_s31 = np.dot(netout, inst[0].T)  # output Sigma31
    (netout_U, netout_S, netout_s, netout_V) = svd(netout_s31)  # s31 = dot(U, S, V.T)
    for i in xrange(len(netout_s)):
        experiment_strength.append(
            [netout_s[i]]
        )

    # 現在の誤差
    err.append(calculate_error(netout, inst[1]))


    for epoch in xrange(n_epochs):
        if epoch%(n_epochs/50) == 0: 
            print epoch, "/", n_epochs
        dW0bar = np.dot(W1bar.T, (S - np.dot(W1bar, W0bar)))  # 式(4)-L
        dW1bar = np.dot((S - np.dot(W1bar, W0bar)), W0bar.T)  # 式(4)-R
        #if DEBUG: print "dW0bar (shape:", dW0bar.shape, ")\n", dW0bar
        #if DEBUG: print "dW1bar (shape:", dW0bar.shape, ")\n", dW1bar

        W0bar += mu*dW0bar
        W1bar += mu*dW1bar
        #if DEBUG: print "W0bar (shape:", W0bar.shape, ")\n", W0bar
        #if DEBUG: print "W1bar (shape:", W0bar.shape, ")\n", W1bar

        # 実際に更新
        W0 = np.dot(W0bar, V.T)
        W1 = np.dot(U, W1bar)
        network.set_W([W0, W1])

        # 理論のstrength (t = epoch + 1)
        for i in xrange(len(s)):
            theory_strength[i].append(
                calculate_strength(t = epoch+1, s = s[i], a0 = a0, tau = 1.0/mu)
            )

        # 実際のstrength(t = 0)
        netout = network.run(inst[0])
        netout_s31 = np.dot(netout, inst[0].T)  # output Sigma31
        (netout_U, netout_S, netout_s, netout_V) = svd(netout_s31)  # s31 = dot(U, S, V.T)
        for i in xrange(len(netout_s)):
            experiment_strength[i].append(
                netout_s[i]
            )

        # 現在の誤差
        err.append(calculate_error(netout, inst[1]))



    W0 = np.dot(W0bar, V.T)
    W1 = np.dot(U, W1bar)

    for elem in theory_strength:
        write_list_to_file(theory_filename, elem)

    for elem in experiment_strength:
        write_list_to_file(experiment_filename, elem)

    write_list_to_file(err_filename, err)


    # 学習出来てるか，試しに走らせてみる
    if DEBUG: print "test run network after learning"
    network.set_W((W0, W1))
    output = network.run(inst[0])
    if DEBUG: print "output(shape:", output.shape, "):\n", output

    # 誤差の計算
    err = calculate_error(output, inst[1])
    if DEBUG: print "final err(output, inst[1]):", err

def create_instruction_signals(n_in, n_out, depth, p, rng = np.random):
    # 例題の入力を作成
    t_input = np.identity(n_in)
    if DEBUG: print "create t_input:\n", t_input

    # 例題の出力を作成
    roots = []
    for i in xrange(n_out):
        if rng.rand() > 0.5:
            roots.append(1)
        else:
            roots.append(-1)

    if DEBUG: print "create BinTree"
    bt = BinTree(depth = depth, p = p, parents = roots, rng = rng)
    if DEBUG: print "created BinTree as bt"

    #t_ans = bt.make_dataset()  # ちがうかも！！
    t_ans = np.array(bt.make_all_tree())
    if DEBUG: print "create t_ans:\n", t_ans

    return (t_input, t_ans)

def svd_reduce(mat):
    U, s, V = np.linalg.svd(mat, full_matrices=False)
    S = np.diag(s)
    #if DEBUG: print "svd is allclose:", np.allclose(mat, np.dot(U, np.dot(S, V)))
    return (U, S, s, V.T)

def svd_full(mat):
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    S = np.zeros((U.shape[1], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    return (U, S, s, V.T)

def calculate_error(mat1, mat2):
    mat = mat1 - mat2
    return (mat*mat).mean()  # これは平均二乗誤差．論文はこれの例題数倍．

def calculate_strength(t, s, a0, tau = 200):  # tau = 1.0/0.005
    try:
        return Decimal(s)*np.exp(Decimal(2*s*t/tau))/(np.exp(Decimal(2*s*t/tau)) - Decimal(1.0) + Decimal(s/a0))
    except InvalidOperation:
        print "t: ", t
        print "s: ", s
        exit(-1)

def write_list_to_file(filename, l):
    fp = open(filename, "a")
    for elem in l:
        fp.write(str(elem))
        fp.write("\n")
    fp.write("\n")
    fp.write("\n")
    fp.close()

if __name__ == "__main__":
    print >> sys.stderr, "start run()"
    run()
    print >> sys.stderr, "done run()"
