# Reproducing McGurk with PerceiverIO
## Using language modeling
### First idea : 
use MaskedLM. it completes a sentence. We may ask give it as input "this person is saying 'da'", mask the last 4 (or 5 with space) tokens, also provide the video, and ask it to input text.