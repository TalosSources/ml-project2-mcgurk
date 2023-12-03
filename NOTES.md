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
## update
using audio only, video only, and audiovisual training examples, the model can still learn a relatively good model. It strangely fails often with video and audio only, but almost never with a combination. When testing mcgurk, it very confidently again assigns it to the visual input. So, it shows a visual domination. The results would be more interesting if we had better accuracy, right now it's still not great with audio inputs only, so we have no strong reason to assume it prefers video, it may simply not be able to recognize audio inputs that well at all.
However, interesting thing. When testing with ba+fa=va, we have this : showing only the fa video predicts fa with almost 100% probability, ba and va being at 0 or e-30 or so. But when adding the ba audio, the probabilities for ba and va become substantially higher, with va being higher than va and reaching 9 * e-03, which is about 1% probability. Not negligible at all, compared to e-30. So, the model was kind of nuged towards the mcgurk thing. It believes the mcgurk hypothesis more than the audio one, but less than the visual one. 
More ideas to improve resutls :
* use all or more of the latents instead of using their mean. We could compute K means, where K divides n_latentS=768, and use a flattened K * 512 vector as X feature. maybe there's more audio explanatory information to be found there.
* 


# Dataset
For N people,
We each say M words
Let's assume F words per second, T seconds.
Then, we know total words $W = T * F * N = M * N$, and also $T * F = M$.
Assume we want $W = 1000$ words, we say $F = 1$ word per second, and we're $N = 5$ people
then $T = W / (F * N) = 1000 / (1 * 5) = 200$. -> so we each need to spend $200$ seconds working on that + inefficiencies.
I'm pretty sure 15 min per person is way enough.


# old ideas

## Using language modeling
### First idea : 
use MaskedLM. it completes a sentence. We may ask give it as input "this person is saying 'da'", mask the last 4 (or 5 with space) tokens, also provide the video, and ask it to input text.
