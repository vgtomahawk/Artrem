#withoutadversary: CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --optimizer=adam --outputmodelname=normalTwo | tee baselineLogFullPrecisionAdam2

#with adversary: CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --use_adv --lambda_adv=0.001 --enc_lstm_dim=256 --outputmodelname=adverseAdam0.01 --optimizer=adam | tee adversaryLogAdam_0.001
