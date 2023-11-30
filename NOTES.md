# Reproducing McGurk with PerceiverIO
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
