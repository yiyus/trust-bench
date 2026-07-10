∇res←Solve req;f;calls;fd;lower;cfg;r;finalpair;grad;status
  f←NameFor req.problem_id
  :If 0=≢f
      res←ErrorResult'Unknown problem_id: ',req.problem_id
      :Return
  :EndIf
  calls←0
  fd←0
  :If 2=⎕NC'req.derivative_mode'
      fd←req.derivative_mode≡'finite-difference'
  :EndIf
  lower←{calls+←1 ⋄ result←Apply f ⍵ ⋄ fd:1⊃result ⋄ result}
  cfg←⎕NS''
  cfg.loss←{2=⎕NC'req.loss':req.loss ⋄ 'L2'}⍬
  cfg.toli←{2=⎕NC'req.max_iter':req.max_iter ⋄ 1E3}⍬
  cfg.tolc←{2=⎕NC'req.tolerance':req.tolerance ⋄ ⎕CT}⍬
  cfg.tolr←{2=⎕NC'req.tolerance':req.tolerance ⋄ ⎕CT}⍬
  :If 2=⎕NC'req.bounds'
      cfg.lower←1⊃req.bounds
      cfg.upper←2⊃req.bounds
  :EndIf
  r←lower Min(req.x0)cfg
  calls←calls+1
  finalpair←Apply f r.p
  grad←(⍉2⊃finalpair)+.×1⊃finalpair
  :If r.iter≥r.toli
      status←'MAX_ITER'
  :ElseIf r.dnorm>r.dmax
      status←'FAILED'
  :Else
      status←'CONVERGED'
  :EndIf
  res←⎕NS''
  res.problem_id←req.problem_id
  res.status←status
  res.message←NULL
  res.x_final←r.p
  res.cost_final←r.cost
  res.n_iter←r.iter
  res.n_feval←calls
  res.n_jeval←NULL
  res.n_heval←NULL
  res.grad_norm_final←0.5*⍨+/grad×grad
∇
