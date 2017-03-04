import tweepy
import auth_token

auth = tweepy.OAuthHandler(auth_token.consumer_key,
                           auth_token.consumer_secret)
auth.set_access_token(auth_token.access_token,
                      auth_token.access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
        print tweet.text

