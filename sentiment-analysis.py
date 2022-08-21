import torch
import pandas as pd
import logging
import os

from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from datetime import datetime

def main(): 

    # read data
    df = pd.read_csv('hotel-reviews.csv')  # only 500 reviews as a sample
    print('data loaded')
    logging.info('data loaded')

    # this is the pre-trained model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

    MODEL = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True)
    
    # if the repo is private, you must get a token from HuggingFace and run `huggingface-cli login` with token
    # then use the following two lines instead
    
    #tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)
    #model = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=True, use_auth_token=True)

    logging.info('model loaded')
    print('model loaded')


    save_n = 100  # save every n rows
    send_email_m = 200  # notify every m rows
    i = 0

    for index, row in df.iterrows():
        i += 1
        review = str(row['review'])[:500]  # only get the first 512 - tensor limit
        output = model(torch.tensor([tokenizer.encode(review)]))
        
        sentiment = int(torch.nn.functional.softmax(output.logits, dim=-1).detach().numpy().argmax())
        sentiment_prob = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0]  # is a list
        print(i, index, review, sentiment, sentiment_prob)
        df.loc[index, 'sentiment'] = sentiment
        df.loc[index, 'sentiment_prob_0'] = sentiment_prob[0]
        df.loc[index, 'sentiment_prob_1'] = sentiment_prob[1]
        
        if i % save_n == 0:  # save to csv every n rows
            df.to_csv('hotel-reviews-processed.csv', index=False)
            print(f'saved to csv at {i} iteration')
            logging.info(f'saved to csv at {i} iteration')
        
        if i % send_email_m == 0:  # send an email every m rows
            send_email(i, 'Ongoing')
    
    # a final save and send an email after finish every thing
    df.to_csv('hotel-reviews-processed.csv', index=False)
    print(f'final save to csv at {i} iteration')
    logging.info(f'final save to csv at {i} iteration')
    send_email(i, 'Completed')


def send_email(current_row, status):

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=f'Sentiment Analysis Processing - {status}',
        html_content=f"<p>the current row is {current_row} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} </p>")
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message) 


if __name__ == '__main__':

    # !huggingface-cli login first with token if repo is private
    # create .env file with sendgrid environment variables
    # load environment variables from .env file

    load_dotenv()
    SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
    FROM_EMAIL = os.getenv('FROM_EMAIL')
    TO_EMAIL = os.getenv('TO_EMAIL')

    # logging
    logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
    
    main()