∇Loop;line;req;res;raw;txt;out
  :Repeat
      line←⍞
      :If 0=≢line
          :Leave
      :EndIf
      :Trap 0
          raw←⊃⎕NGET line 1
          txt←∊raw,¨⎕UCS 10
          req←⎕JSON txt
          :If 2=⎕NC'req.mode'
          :AndIf req.mode≡'evaluate'
              res←Evaluate req
          :Else
              res←Solve req
          :EndIf
      :Else
          res←ErrorResult ⎕DMX.EM,': ',⎕DMX.Message
      :EndTrap
      out←⎕JSON⍠'HighRank' 'Split'⊢res
      ⍝ ⎕← wraps any single output past 32767 characters into several
      ⍝ physical \r-terminated segments (confirmed directly: a large
      ⍝ Hessian at dimensionality(n=1000) triggers this) - the length
      ⍝ is announced on its own (always short) line first so the reader
      ⍝ knows how many payload characters to expect regardless of how
      ⍝ many physical lines they arrive wrapped across.
      ⎕←≢out
      ⎕←out
  :EndRepeat
∇

Loop
