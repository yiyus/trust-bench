∇Run paths;input;output;raw;txt;req;res
  input←1⊃paths
  output←2⊃paths
  :Trap 0
      raw←⊃⎕NGET input 1
      txt←∊raw,¨⎕UCS 10
      req←⎕JSON txt
      res←Solve req
  :Else
      res←ErrorResult ⎕DMX.EM,': ',⎕DMX.Message
  :EndTrap
  (⎕JSON res)⎕NPUT output 1
  ⎕OFF('ERROR'≡res.status)
∇

Run 1↓2 ⎕NQ '.' 'GetCommandLineArgs'
