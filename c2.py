#!/usr/bin/python

import numpy as np
import random

CODES0 = [
  25,
  50,
  75,
  100,
  125,
  150,
  175,
  200,
  225,
  250,
  275,
  300,
  325,
  350,
  375,
  400,
  425,
  450,
  475,
  500,
  525,
  550,
  575,
  600,
  625,
  650,
  675,
  700,
  725,
  750,
  775,
  800
]
CODES3 = [
  25,
  50,
  75,
  100,
  125,
  150,
  175,
  200,
  250,
  300,
  350,
  400,
  450,
  500,
  550,
  600,
  650,
  700,
  750,
  800,
  850,
  900,
  950,
  1000,
  1050,
  1100,
  1150,
  1200,
  1250,
  1300,
  1350,
  1400
]
CODES = (CODES0,CODES0,CODES0,CODES3,CODES3,CODES3,CODES0,CODES0,CODES0,CODES0)

COMPONENTS_PER_WORD = 14
U = np.uint64

def degray(field):
  field = field.item() #field.astype(np.uint32)
  t = field ^ (field >> 8);
  t ^= (t >> 4);
  t ^= (t >> 2);
  t ^= (t >> 1);
  return t

def unpack(w, ofs, nb):
  maxval = (1<<nb)-1
  ind = unpackind(w, ofs, nb)
  return ind * 2.0 / maxval - 1.0

def unpack2(w, ofs, nb):
  maxval = (1<<nb)-1
  ind = unpackind(w, ofs, nb)
  return ind * 1.0 / maxval

def unpackind(w, ofs, nb):
  maxval = (1<<nb)-1
  #rot = ofs+nb-((ofs+nb)/8)*8
  #w = ((w >> U(ofs)) & U(maxval))
  #ind = ror(w, U(rot), U(nb)) & U(maxval)
  #ind = degray(ind)
  ind = (w >> U(64-ofs-nb)) & U(maxval)
  ind = degray(ind)
  #print(hex(w),nb,ofs,'=',ind)
  return ind

def rol(x,s,n):
  return (x << s) | (x >> (n-s))

def ror(x,s,n):
  return (x << (n-s)) | (x >> s)

def decodeWord(w):
  vb1 = unpack(w, 0, 1)
  vb2 = unpack(w, 1, 1)
  Wo = unpack(w, 2, 7)
  e = unpack(w, 9, 5)
  lsps = []
  freq = 0
  for i in range(0,10):
    lspcbi = unpackind(w, 14+i*5, 5)
    freq += CODES[i][lspcbi]
    fcen = (i+0.5)*4000/11.0
    lsps.append((freq-fcen)/1000.0)
  return [vb1,vb2,Wo,e] + lsps

def engray(field):
  return (field >> U(1)) ^ field

def pack(w, ofs, nb, x):
  maxval = (1<<nb)-1
  ind = round((x+1)*0.5*maxval)
  return packind(w, ofs, nb, ind)

def pack2(w, ofs, nb, x):
  maxval = (1<<nb)-1
  ind = round(x*maxval)
  return packind(w, ofs, nb, ind)

def packind(w, ofs, nb, ind):
  maxval = (1<<nb)-1
  ind = min(max(ind, 0), maxval)
  gc = engray(U(ind)) & U(maxval)
  w |= (gc << U(64-ofs-nb))
  return w

def quantize(x, codes):
  besti = 0
  bestv = 99999
  for i in range(0,len(codes)):
    v = abs(x-codes[i])
    if v < bestv:
      besti = i
      bestv = v
  return besti

def encodeWord(params):
  assert(len(params) == 14)
  w = U(0)
  w = pack(w, 0, 1, params[0]) # vb1
  w = pack(w, 1, 1, params[1]) # vb2
  w = pack(w, 2, 7, params[2]) # Wo
  w = pack(w, 9, 5, params[3]) # e
  freq0 = 0
  for i in range(0,10):
    fcen = (i+0.5)*4000/11.0
    freq = fcen + (params[4+i])*1000
    df = freq-freq0
    ind = quantize(df, CODES[i])
    freq0 += CODES[i][ind]
    w = packind(w, 14+i*5, 5, ind)
  return w

def decode_c2file(f):
  return np.array([decodeWord(w) for w in np.fromfile(f, dtype='>u8')], dtype=np.float32)

def encode_c2file(f,words):
  arr = np.array([encodeWord(w) for w in words], dtype='>u8')
  arr.tofile(f)

# split phrases by silence (energy < vol for silencelen frames)
def split_phrases(words, vol, silencelen, minlen):
  buf = []
  phrases = []
  n = 0
  for w in words:
    if w[3] < vol:
      n += 1
    else:
      if n >= silencelen:
        if len(buf) >= minlen:
          phrases.append(buf)
          buf = []
        n = 0
      buf.append(w)
  if len(buf) >= minlen:
    phrases.append(buf)
  return phrases

blank_frame = np.array([0.0]*3 + [-1.0] + [0]*10, dtype=np.float32)

# returns a silent frame with noise
def get_silent_frame(noise=0.5):
  f = np.random.randn(14) * noise
  # set energy to 0 (silent)
  f[3] = 0
  return f

###

if __name__ == '__main__':
  print(get_silent_frame())
  # read codec2 3200 bps file
  with open('obama.c232','rb') as f:
    words = decode_c2file(f)
    
  print(np.mean(words, axis=1))
  print(np.std(words, axis=1))

  # shuffle phrases?
  if 0:
    phrases = split_phrases(words, 0.2, 10, 10)
    print ( [len(ph) for ph in phrases] )
    import random
    random.shuffle(phrases)
    words = []
    for ph in phrases:
      words.extend(ph)

  # add noise?
  if 0:
    for i in range(len(words)):
      words[i] = np.random.randn(14)/32.0 + words[i]

  # generate codec2 3200 bps file
  with open('test.c232','wb') as f:
    encode_c2file(f,words)
