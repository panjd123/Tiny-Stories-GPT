python gpt.py --batch-size 256 --embedding-dim 192 --num-heads 6 --num-layers 4 --tokenizer tinystories_bpe_tokenizer/4K
python gpt.py --batch-size 256 --embedding-dim 384 --num-heads 12 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/16K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer gpt2
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer gpt2
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer gpt2

python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/50K --max-iterations 100 --num-stories 6000
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer gpt2 --max-iterations 100 --num-stories 6000