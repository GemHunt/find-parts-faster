# find-parts-faster
**Second Prototype MileStone by 2/15/18:**
This is a non-robotic system to find similar looking, but rare small parts. The focus will be on fasteners, but generic functions will handle all kinds of small parts. Parts will go into a tray on computer display with a camera above it. 

**What will this do:**
* It’s quick solution for something not worth the time and effort of automation
* ID: What type of fastener is this? What type of thread?
* QA: Find rare bad parts 
* Search: Finding the last parts that are really needed.
* Manual sorting: Raises quality and flexibility with a little speed gains.

**Usage:**
* Put parts on the the tray
* Shake the tray a bit to separate them
* It’s ok if they are touching, but they can’t be stacked
* Parts will be highlighted by the display underneath if they need to be picked

**Speed Gains by using the system:**
Search speed will depend on the rareness of the parts and visual difference. A great example is rarity of one in one hundred. It might take someone 1 second a part vs 6 parts a second using the system. I have worked with coins, screws and legos so to coin up with that number. 

**With 99.99% accuracy:**
* It will find parts that don’t belong 
* It will find burrs, damaged, or unfinished areas ½ of a thread diameter and up. 
* Users can train their own items to classify as a set(the normal way) 
* Users can self-train their own items to classify as one. (train this part to itself)
* It’s not a sorting system, but it can speed hand sorting, by verifying the result
* It’s not a precise measuring system
* It will ID all common threads from about 4 to ⅜, 2mm-10mm, machine, wood, sheet metal, etc
* You can get an inventory of parts on trays(just as a demo) 
* Lengths on non-touching screws

**99.99% accuracy applies to sizes larger than 1/3rd of a thread diameter:**
* For example 
* It will disciminate between 9 and 10mm long M3 screws
* It will notice a 1mm x 1mm area of crushed M3 thread
* It will highlight a 2mm x 5mm chip burr on a M6 screw

**What it does with 99% accuracy on first pass:**
* 99% will slow people down, but it’s still very useful 
* Basically this means you will need to jiggle the tray more that once
* The operator might find part before the system does
* Lengths on non-touching screws 
* It will find parts and areas of parts that just don’t match as well(for whatever reason) 
* Internal thread ID

**First proof of concept history:**
* I build the first version in summer 2017
* I proved out this setup physically works(the ergonomics of it)
* It works and it's fun it use
* Used traditonal CV functions
* But... it will not scale due to touching and training issues





