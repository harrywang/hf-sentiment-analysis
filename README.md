# hf-sentiment-analysis
Code for my tutorial at https://harrywang.me/jupyter

Create a `.env` file to store the environment variables the program needs - setup the SendGrid API key and from/to emails in our SendGrid account.

```
SENDGRID_API_KEY='SG.t1o-xxxx'
FROM_EMAIL='your_from_email'
TO_EMAIL='your_to_email'
```

Optional: if stuck at `Building wheel for tokenizers (pyproject.toml) ..` during `pip install..` follow the instruction [here](https://github.com/huggingface/transformers/issues/2831#issuecomment-1001437376) for Mac M1.

```
brew install rustup
rustup-init
source ~/.cargo/env
rustc --version
pip install tokenizers
```
tokenizers installation step may take a while to finish - just be patient.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python sentiment-analysis.py
```
