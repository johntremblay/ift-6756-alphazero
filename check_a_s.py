from santorini.SantoriniGame import SantoriniGame

g = SantoriniGame(5,4)

import json

dict = g.get_all_action_move()

json = json.dumps(dict)
f = open("dict.json","w")
f.write(json)
f.close()