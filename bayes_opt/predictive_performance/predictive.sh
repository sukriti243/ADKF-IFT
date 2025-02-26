#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate fsmol

SUPPORT=32
SUPPORTT=64

CUDA_VISIBLE_DEVICES=3, python adkt.py dock $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python dkt.py dock $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python protonet.py dock $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python cnp.py dock $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gnnmt.py dock $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gpst.py dock $SUPPORT $SUPPORTT

CUDA_VISIBLE_DEVICES=3, python adkt.py anti $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python dkt.py anti $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python protonet.py anti $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python cnp.py anti $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gnnmt.py anti $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gpst.py anti $SUPPORT $SUPPORTT

CUDA_VISIBLE_DEVICES=3, python adkt.py covid $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python dkt.py covid $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python protonet.py covid $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python cnp.py covid $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gnnmt.py covid $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gpst.py covid $SUPPORT $SUPPORTT

CUDA_VISIBLE_DEVICES=3, python adkt.py opv $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python dkt.py opv $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python protonet.py opv $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python cnp.py opv $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gnnmt.py opv $SUPPORT $SUPPORTT
CUDA_VISIBLE_DEVICES=3, python gpst.py opv $SUPPORT $SUPPORTT

# CUDA_VISIBLE_DEVICES=3, python mat.py dock $SUPPORT $SUPPORTT
# CUDA_VISIBLE_DEVICES=3, python mat.py anti $SUPPORT $SUPPORTT
# CUDA_VISIBLE_DEVICES=3, python mat.py covid $SUPPORT $SUPPORTT
# CUDA_VISIBLE_DEVICES=3, python mat.py opv $SUPPORT $SUPPORTT
