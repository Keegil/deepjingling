
# coding: utf-8

# In[ ]:

import numpy as np
import os
from cntk.persist import load_model

# Sample from the network
def sample(root, ix_to_char, vocab_dim, char_to_ix, prime_text='', use_hardmax=False, length=300, temperature=1.0):

    # temperature: T < 1 means smoother; T=1.0 means same; T > 1 means more peaked
    def apply_temp(p):
        # apply temperature
        p = np.power(p, (temperature))
        # renormalize and return
        return (p / np.sum(p))

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p, axis=2)[0,0]
        else:
            # normalize probabilities then take weighted sample
            p = np.exp(p) / np.sum(np.exp(p))            
            p = apply_temp(p)
            w = np.random.choice(range(vocab_dim), p=p.ravel())
        return w

    plen = 1
    prime = -1

    # start sequence with first input    
    x = np.zeros((1, vocab_dim), dtype=np.float32)    
    if prime_text != '':
        plen = len(prime_text)
        prime = char_to_ix[prime_text[0]]
    else:
        prime = np.random.choice(range(vocab_dim))
    x[0, prime] = 1
    arguments = ([x], [True])

    output=[]
    output.append(prime)
    
    # loop through prime text
    for i in range(plen):            
        p = root.eval(arguments)        
        
        # reset
        x = np.zeros((1, vocab_dim), dtype=np.float32)
        if i < plen-1:
            idx = char_to_ix[prime_text[i+1]]
        else:
            idx = sample_word(p)

        output.append(idx)
        x[0, idx] = 1            
        arguments = ([x], [False])
    
    # loop through length of generated text, sampling along the way
    for i in range(length-plen):
        p = root.eval(arguments)
        idx = sample_word(p)
        output.append(idx)

        x = np.zeros((1, vocab_dim), dtype=np.float32)
        x[0, idx] = 1
        arguments = ([x], [False])

    # return output
    return ''.join([ix_to_char[c] for c in output])

def load_and_sample(model_filename='models/deepjingling-songwriter4_epoch97.dnn', 
                    vocab_filename='data/songs.txt.vocab', 
                    prime_text='', 
                    use_hardmax=False, 
                    length=500, temperature=1.0):
    
    # load the model
    model = load_model(model_filename)
    
    # load the vocab
    chars = [c[0] for c in open(vocab_filename, encoding='utf8').readlines()]
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
        
    output = sample(model, ix_to_char, len(chars), char_to_ix, prime_text=prime_text, use_hardmax=use_hardmax, length=length, temperature=temperature)
    
    print(output)

