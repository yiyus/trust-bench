∇res←Evaluate req;f;r;j
  f←NameFor req.problem_id
  :If 0=≢f
      res←ErrorResult'Unknown problem_id: ',req.problem_id
      :Return
  :EndIf
  r j←Apply f(req.x)
  res←⎕NS''
  res.problem_id←req.problem_id
  res.status←'OK'
  res.message←NULL
  res.residual←r
  res.jacobian←j
∇
