# PDB_ID=7rvu
# PDB_ID=5x1y
PDB_ID=6o7q

python demo_esm.py --cfg-path configs/evaluation.yaml  --gpu-id 0  --pdb /home/ubuntu/dataset/pt/$PDB_ID.pt --seq /home/ubuntu/dataset/seq/$PDB_ID.pt
