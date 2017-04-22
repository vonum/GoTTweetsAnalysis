def check_for_duplicate(ids1, ids2):
  ids = []

  for id in ids1:
    if id in ids:
      print 'FAILED IN COL1'
      break

    ids.append(id)

  for id in ids2:
    if id in ids:
      print 'FAILED IN COL1'
      break

    ids.append(id)

  print 'FINISHED'
