# Reproducing McGurk with PerceiverIO
## Using AutoEncoding :
We managed to make a very reliable syllable classifier, that can with at least 98% classify syllables between [ba, da, ga], even with videos taken with selfie that have nothing to do with the training data.
We used a logistic regression multi-classifier taking as input the last hidden self-attention latents from the Perceiver model, and optimizing with Adam.
### Testing McGurk
We combined videos gas and audios bas to get ga+ba videos, and fed it to the trained classifier. It answers all the time 'ga' with very high confidence. This suggests a strong preference over video.

# Options to improve what we have and isolate McGurk:
* use a more diverse dataset -> actually, shouldn't change a lot, because the current one is very good already. maybe a diverse one with video being less obvious would force the model to learn some audio features
* train with masked audio, and masked video successively, to force the model to be able to classify using just one of the 2 modalities, giving credit to our assumptions about the result.
* use another model -> olena's exploration direction

## Using language modeling
### First idea : 
use MaskedLM. it completes a sentence. We may ask give it as input "this person is saying 'da'", mask the last 4 (or 5 with space) tokens, also provide the video, and ask it to input text.


# Dataset
For N people,
We each say M words
Let's assume F words per second, T seconds.
Then, we know total words $W = T * F * N = M * N$, and also $T * F = M$.
Assume we want $W = 1000$ words, we say $F = 1$ word per second, and we're $N = 5$ people
then $T = W / (F * N) = 1000 / (1 * 5) = 200$. -> so we each need to spend $200$ seconds working on that + inefficiencies.
I'm pretty sure 15 min per person is way enough.
