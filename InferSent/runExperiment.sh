#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam0.0_addAnnealing_0.1_5e-6_FullThroughAdversary --optimizer=adam --lambda_adv=0.0 --addAnnealing --max_lambda_adv=0.1 --anneal_growth_rate=5e-6  --full_through_adversary  --use_adv  --no_early_stopping | tee adversaryLogAdam_0.0_addAnnealing_0.1_5e-6_fullThroughAdversary

#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.01_FullThroughAdversary --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.01   --full_through_adversary  --use_adv  --no_early_stopping | tee adversaryLogAdam_reversalWeight=0.01_FullThroughAdversary


#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=baselineAdam3 --optimizer=adam  --no_early_stopping | tee baselineAdamLog3

#Train baseline on MultiNLI
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=baselineAdam3_MultiNLI --optimizer=adam  --no_early_stopping --nlipath="dataset/MultiNLI/" | tee baselineAdamLog3_MultiNLI

#Train adversary on MultiNLI
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.01_MultiNLI --optimizer=adam  --no_early_stopping --lambda_adv=1.0 --reversal_weight=0.01 --use_adv  --nlipath="dataset/MultiNLI/" | tee adversaryLogAdam_reversalWeight=0.01_MultiNLI


#Train adversary
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.01 --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.01     --use_adv  --no_early_stopping | tee adversaryLogAdam_reversalWeight=0.01

#Train adversary with lower coefficient
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.001 --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.001     --use_adv  --no_early_stopping | tee adversaryLogAdam_reversalWeight=0.001


#Train deeper adversary
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.001_deeperAdv --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.001     --use_adv  --no_early_stopping --deeper_adv | tee adversaryLogAdam_reversalWeight=0.001_deeperAdv

CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.01 --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.01     --use_adv  --no_early_stopping  --evaluateOnly --evalExt --extNlipath=dataset/stress_tests/Swap/


#Test the model
#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=adverseAdam_reversalWeight=0.1 --optimizer=adam --lambda_adv=1.0 --reversal_weight=0.1    --use_adv  --evaluateOnly --evalExt --extNlipath=dataset/HardNLI/

#CUDA_VISIBLE_DEVICES="" python -u train_nli.py --word_emb_path dataset/GloVe/glove.840B.300d.txt --enc_lstm_dim=256 --outputmodelname=baselineAdam3 --optimizer=adam  --no_early_stopping --evaluateOnly --evalExt --extNlipath=dataset/stress_tests/Swap/

