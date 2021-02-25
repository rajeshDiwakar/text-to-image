""" from https://github.com/keithito/tacotron """
import re
import random
from text import cleaners
from text.symbols import symbols

import numpy as np


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def get_arpabet(word, dictionary):
  word_arpabet = dictionary.lookup(word)
  if word_arpabet is not None:
    return "{" + word_arpabet[0] + "}"
  else:
    return word


def text_to_sequence(text, cleaner_names, dictionary=None, p_arpabet=1.0,aligndict=None, p_align=0,dim_align=0):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  space = _symbols_to_sequence(' ')
  # Check for curly braces and treat their contents as ARPAbet:
  alignments = []
  align_zero = np.zeros((dim_align,1))
  # lengths = []
  while len(text):
    m = _curly_re.match(text)
    if not m:
      clean_text = _clean_text(text, cleaner_names)
      if cmudict is not None:
        clean_text = [get_arpabet(w, dictionary)
                      if random.random() < p_arpabet else w
                      for w in clean_text.split(" ")]

        for i in range(len(clean_text)):
            t = clean_text[i]
            if t.startswith("{"):
              element = _arpabet_to_sequence(t[1:-1])
              if dim_align>0:
                  alignments.extend([align_zero for _ in range(len(element))])
            else:
              element =  _symbols_to_sequence(t)
              if dim_align>0:
                  if random.random()< p_align:
                      try:
                          word_align = aligndict[t.strip().lower()][2]
                          temp_align = np.zeros((dim_align,len(element)))
                          r,c = word_align.shape
                          assert c==len(element),'text:%s, enc:%s ,cols:%d'%(t,str(element),c)
                          r = min(r,dim_align)
                          temp_align[:r,:] = word_align[:r,:]
                          alignments.append(temp_align)  # align shape 94x8
                      except KeyError:
                          alignments.extend([align_zero for _ in range(len(element))])
                  else:
                      alignments.extend([align_zero for _ in range(len(element))])
            sequence += element
            sequence += space
            if dim_align>0:
                alignments.append(align_zero)
            # lengths.append((len(sequence),sum([i.shape[1] for i in alignments])))
      else:
        sequence += _symbols_to_sequence(clean_text)
      break

    clean_text = _clean_text(text, cleaner_names)
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # remove trailing space
  if  sequence[-1] == space[0]:
      sequence = sequence[:-1] #if sequence[-1] == space[0] else sequence
      alignments = alignments[:-1] #if sequence[-1]==space[0] else alignments
  alignments = np.concatenate(alignments,axis=1) #time x text
  assert len(sequence)==alignments.shape[1],'\n\n'+str(lengths)+'\ntext length is different from alignments\nlen(sequence):%d, align shape:%s'%(len(sequence),str(alignments.shape))
  return sequence,alignments


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
