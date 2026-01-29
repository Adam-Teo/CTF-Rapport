
# Capture The Flag
#### - Tillämpning Av Reinforcement Learning i Spel


## Problem 
**Fråga A:** Hur kan *Reinforcement Learning* implementeras för att träna en agent att spela *Capture the Flag*. 

**Fråga B:** Hur kan hyperparametrar implementeras vid träning för att förbättra agenten dvs få den att spela mer strategiskt och effektivt.


## Capture the Flag
#### Definition av *Capture the Flag*
I den här projektet definers *Capture the Flag* (*CTF*) på följande vis: Ett spel där två lag tävlar om
att vara först med att plocka upp motståndarlagets flagga och föra den tillbaka till sin bas. 



#### Objekt Index

![image](img/Object_Index_Wall_Floor.png)
*Wall                            Red Floor                    Blue Floor*

![[Object_Index_Player.png]]
*Red Player                  Blue Player                  Red Flag Carrier         Blue Flag Carrier*

![[Object_Index_Flagg_Base.png]]
*Red Flag Base             Blue Flag Base            Empty Red Base           Empty Blue Base*                         


#### Hur CTF fungerar
Ett lag vinner genom att plocka upp motståndarlagets flagga och föra den tillbaka till sin bas. 

![[Flag_Score.gif]]
*Spelare blå tar det röda lagets flaggan tillbaka till sin egen bas och vinner spelet*

###### Tagging
Motståndare laget kan ta tillbaka sin flagga om de ockuperar samma position som den flaggbärande motståndaren. Detta sänder flaggan tillbaka till sin hem bas och den flaggbärande spelaren tillbaka till sin start punkt.

![[Defensive_Action.gif]]
*Röd laget skyddar sin flagga genom att tagga de blåa spelarna som tar deras flaggan*

###### Spelplan
Spelplanen består av ett grid på 17x17 där varje koordinat representerar ett objects position. 
Väggar och spelarnas startpositioner är slumpvist utplacerad på spelplanen för att göra det svårare för spelarna att ta sig direkt till motståndare lagets flagga, detta tvingar agenten an hitta unika lösningar för varje match. 

Värt att notera är att dessa väggar speglas på båda sidorna så varje spelhalva är symmetriskt jämställde så inget lag har fördel pga spelplanens utformning, dock kan de få en liten fördel pga startpositionerna som inte är speglade. flaggorna och baserna där flaggorna befinner sig dock alltid på samma ställe för att ge agenten en fast punkt att röra sig mot. 

![[Mirrored_Walls.gif]]
*Exempel på olika framslumpade spelplaner*

###### Synfält
Spelarna ser bara en del av spelplanen, deras synfält är består av sex rutor framåt och tre åt vardera sida dvs 7x7. Detta blir en input bild som konverteras till en tensor och skickas till CNN som beräknar vilken typ av handling som kommer leda till störst reward för spelaren.

Agenten kan utföra en av dessa tre handlingar varje steg; rotera 90° vänster, rotera 90° höger, gå framåt.
Det finns även tre indirekta handlingar som sker automatiskt när agenten flyttar en spelaren till samma koordinat som ett specifikt object; plocka up en flagga, lämna in flaggan, tagga en fiende spelare som bär en flagga. 

![[Player_Vision.png]]
*Den gula kanten visar vad den röda spelaren ser,  Triangeln visar den röda spelarens position, notera att den röda spelaren inte ser triangle dvs den ser inte sig själv*

#### Agent och Spelare
Med agent syftar detta projektet på vikterna som styr beslutsfattningen. Med spelare syftar till de fyra olika spelarna uppdelade i de två lagen. Det är agenten som styr varje spelare, agenten tittar i tur och ordning igenom spelarens ögon dvs den tar en 84x84 bild och använder som input till CNN och flytta eller roterar spelaren beroende på vilken av de tre handlingar den som CNN beräknar kommer leda till störst reward. 

En agenten styr alla fyra spelare men det är viktigt att notera att agenten alltid spelar utifrån den nuvarande spelares synvinkel och intresse. Agenten tar ett beslut som den tror kommer maximerar reward för spelaren, inte för spelet i sin helhet, den kommer inte ihåg vad de andra spelarna precis gjorde och på så sätt så isoleras spelarna i från varandra och effekten blir att varje spelare styrs av en egen agent även om de delar samma vikter.

Detta är intressanta konsekvens av arkitektur, det är mycket möjligt att vissa vikterna endast aktiveras för vissa spelare och på så sätt skapar ett unikt nätverk för varje spelare. Hade mer tid funnits hade det varit intressant att göra en mer djupgående EDA för att analysera och se om olika vikter aktiveras för olika spelare.


## Moduler & Verktyg
Dessa är verktygen som används för att besvara Fråga A: *Hur kan Reinforcement Learning implementeras för att träna en agent att spela Capture the Flag* 

###### Gymnasium
*Gymnasium* modulen är en popular *Python* baserat *Reinforcement Learning* biblioteket.  *Gymnasium* utgör grundstommen i det här projektet då det hanterar det mesta som sker under ytan. *Gymnasium* gör följande; Hanterar game loopen med `step()`, initierar spelplanen med `reset()`,  hantera  de tre handlingar som agent kan ta med `Descret(3)`. Och  definierar hur inputen ska se ut via `Box()` dvs hur många färg kanaler, högsta lägsta pixel värdet, bildstorlek, bild format osv. 

###### MiniGrid
*MiniGrid* modulen bygger på *Gymnasium* och används för att skapa och rendera den 2-dimensionella spelvärld som agenten existerar i. Den tar den abstract numeriska 17x17 representationen och skalar upp den så varje koordinat blir en 12x12 pixel stor ruta. Dessa 12x12 stora pixel rutor är vad agenten ser. *MiniGrid* kommer med färdig definierad object som väggar, golv, spelare, dörrar osv. Och den skapar även regler för hur spelarna navigerar igenom miljön. 

###### SuperSuit
*SuperSuit* är en hjälpfunktion som kontrollerar at datan från *MiniGrid* matchar den definierade `Box()` funktionen i *Gymnasium*. Är datan i fel format så korrigeras den av *SuperSuit* så att *PPO-Algoritmen* inte kraschar. Kort sagt *SuperSuit* plattar till ut datan och ser till att den följer ett standardiserat format så *StabelBaseline3* kan hantera den. Den hanterar även *Frame Stacking* dvs den sammanfogar tre bilder till en sekvens för att ge context till inlärningen.

###### StableBaselines3
Hanterar *PPO-Algoritmen* dvs den hanterar de neural nätverket, gradient och uppdaterar vikterna under träning. Den tar in observationerna och rewards och tar beslut om vilka vikter som ska uppdateras.

###### PettingZoo
*MiniGrid* och *Gymnasium* kan inte hantera multi-agent scenarion, *PettingZoo* tar hand om den biten, den håller koll på vilken spelares tur det är och ser till att varje spelare får rätt reward. 

#### Data Flow
*PettingZoo (Logiken/Laget) $\rightarrow$ *MiniGrid (Världen/Grafiken) $\rightarrow$ *SuperSuit (Filtret/Stacking)* $\rightarrow$ StableBaselines3 (PPO/Hjärnan)*
	
## Träning 
Denna del beskriver hur *Reinforcement Learning* processen fungerar och ger svar på  Fråga A: *"Hur kan Reinforcement Learning implementeras för att träna en agent att spela Capture the Flag"* 

1) Pre-Game
	`reset()` funktionen genererar ett  17x17 gird med object, vissa av dessa som spelar positioner och väggar placeras ut slumpmässigt. Detta är en abstract numerisk representation av världen som kommer användas för att rendera grafiken. Spelarna får även roller tilldelade baserat vem som är närmast sin egen flagga.

2. Observation
	Agenten tittar igenom varje spelares synfält och samlar in datan i en tensor form som representerar en bild som är 84 pixlar hög 84 pixlar bred och har 3 färg kanaler. Denna bild är en grafisk representation av en 7x7 bit av den 17x17 stora grid världen som agenten kan se. 
	
	Anledningen till varför vi använder en *CNN* och inte bara skickar den råa datan direkt från *MiniGrid* till vårt *Neurala Nätverk* är för att simulera ett verkligt scenario där agenten inte har tillgång till källkoden utan, likt en människa, bara kan se en visuell representation av spelet i form av en serie bilder. Fördelen är att  agenten blir mer generell och att den, i teorin, kommer kunna tränas på liknande spel utan att behöva ändra den grundläggande arkitekturen. Nackdelen är att träningen tar längre tid då den behöver behandla en större mängd information.

3) Input
	*Frame Stacking* används för att ge träningen mer kontext genom att stacka 3 frames på varandra. De stackade frames kan sees som en multiplication av färg kanaler dvs 3 frames x 3 färg kanaler skapar 9 dimensioner, vilket gör inputen till 84x84x9. 

	För att snabba up träningen används *Multi Environment*  dvs 4 environments tränas parallellt, med varandra. Eftersom vardera environment innehåller 4 spelare så får vi 4x4, totalt 16. Så den slutgiltiga batch sizes är 84x84x9x16. 

	Skillnaden mellan *Frame Stacking* och *Multi Environment* är att *Frame Stacking* ger mer kontext, dvs den slår ihop datan medans *Multi Environment* separerar datan så den kan behandlas parallellt.  

	En binär variable skickas också med som representerar spelarens roll, *Attacker* eller *Defender*, så den slutgiltiga inputen är 16 batcher av en tensor med storlek 84x84x9 som skickas till CNN och en binär variabel som skickas vid sidan av till en MLP dessa kopplas sedan samman inuti nätverket.

	Värt att notera är hyperparametern`n_steps`, detta är en buffer som styr hur mycket data som samla sin dvs hur många steps som går innan vikterna uppdateras. 
. 
4) Handlingar
	*PPO-algoritmen* dvs det *Neurala Nätverket* behandlar input datan och får ut en logits för varje handling som beskriver hur "bra" den tror tre olika handlingarna är. 
	
	Vid Inference dvs om agenten inte tränas utan spelar en testmatch så väljer agenten den handling som har högst sannolikhet att ge störst reward dvs har högst logit värde. 
	
	Under träning sker samma process men agenten har även en *Entropy Coefficient* styr hur nyfiken agenten är vilket göra att den kan ignorera logit värdet och istället utföra en mindre optimal handling, detta för att den inte ska fastna i ett mönster dvs ett lokalt minimum. Genom att get agenten möjlighet att utforska miljön med en större variation av handlingar så har den en större chans att lära sig nytt och förbättra sina strategier, detta kallas *Exploration*.

	Efter att en handling har valts för varje spelar så behöver de kontrolleras för att se om de är giltiga drag.  Om handlingen inte är giltigt ex om spelaren försöker gå igenom en vägg så kommer inte handlingen utföras, spelaren förlora sitt drag och står still. 
	Notera att en hyperparameter har lagt till i som en *penalty* för att straffa agenten när den går in i en vägg.
	
	När det bara är giltiga handlingar kvar så utförs dom samtidigt dvs om spelare röd_1 rör sig in i ruta 8x8 och spelar blå_0 rör sig ut ur ruta 8x8 med flaggan så kommer spelare blå_0 **inte** att bli taggad. Båda spelarna måste stå på samma ruta efter att alla handlingar utförts för att en av dem ska kunna bli taggad. 

	Efter att alla actions utförts så uppdateras världen och de nya positionerna sparas. 
	
5) Reward
	*MiniGrid* skickar tillbaka rewards till *PPO-algoritmen* där delas detta upp i två delar, *Aktör* och *Kritiker*.

	*Kritikern* jämför det faktiska utfallet med det förväntade utfallet och räknar ut  *Advantage* som är mellan skillnaden på dessa två tal. 

	*Aktören* får *Advantage* talet från *Kritikern*, om den är positive dvs om handlingen är bättre än förväntat så uppdateras vikterna så att en sådan action blir mer sannolik. Är resultatet negativts så ändras vikterna så den handlingen blir mindre sannolik och om *Advantage* är noll dvs förutsägelsen stämmer överens med verkligheten så behöver inte vikterna justeras.  

6) End State
	Det finns tre möjlig *end states*:
	 
	A: Den sammanlagda mängde steg överskrider stop-villkors gränsen så avslutas träningen och agenten dvs vikterna sparas till disk.

	 B: Om en handling eder till att ett lag vinner eller om max stegen för spelomgången har uppnåtts så är spelet slut om tränings processen fortsätter från steg 1
	
	C: Spelomgången är inte avslutad och tränings processen fortsätter från steg 2


## Hyperparametrar 
Träningsmodellen har tre klasser av hyperparametrar; Environment, Meta och Rewards. Hur dessa implementeras och används ger svar på Fråga B: *"Hur kan hyperparametrar implementeras vid träning för att förbättra agenten dvs få dem att spela mer strategiskt och effektivt."*

#### Environment
Environment parametrarna har att göra med hur grid miljön är uppbyggd samt spelets längd.

###### grid_size 
Spelplanens storlek, till skillnad från ett spel som schack där spelarens synfält täcker hela brädet så har spelarna i denna version av *CTF* en bestämd storlek vilket innebär att storleken på spelplanen kan varieras utan att påverka inputen till det neurala nätverket, detta gör spelplanen till en hyper parameter, en agent kan tränas på en liten plan och spela på en större variant.  Att träna en agent på en liten plan kan vara till fördel då den har lättare att hitta till motståndarens flagga. 

###### mirrored_walls
Denna parameter styr hur många framslumpade vägar som placeras ut, ju fler väggar ju svårare blir det för spelarna att navigera miljön. 

###### center_walls
Bestämmer hur många väggar ska fylla mitten raden av spelplanen

###### max_step
Bestämmer hur många steg (drag) spelarna får göra innan spelet tar slut, hög *max step* gör att spelarna får mer tid till att hitta till flaggan men den kan potentiellt sakta ner träningen om spelarna fastnar så tar det längre tid för spelet att komma till en `reset()`.

#### Meta
Dessa parametrar styr själva tränings processen. 

###### learning_rate
Styr hur mycket vikterna korrigeras under tränings processen, låg *learning rate* inbära att vikterna blir mer finjusterade men kan ha svårare att ta sig ur lokal minimum.
###### ent_coef
I början av träningen så slumpas alla viker fram och sedan justeras dess vikter under träningens gång. *Entropy Coefficient* styr hur nyfiken agenten är,  en hög `ent_coef` innebär att agenten vilja variera handlingar under träningen.   
###### batch_size
Avgör hur stora data chunks som behandlas på samma gång, med en stark GPU klarar av större batch sizes vilket ökar träningshastigheten.

##### n_steps
Hur stor buffert är dvs hur mycket data som samlas in dvs hur många steps som går innan spelet pausas och träningen drar igång

###### num_vec_envs
Hur många environments dvs agenter som kan tränas parallellt.

## Rewards 
Dessa hyperparametrar styr spelarnas beteenden.

#### Grundläggande Rewards

###### REWARD_TAG_ENEMY
Spelarna får en reward när de tar tillbaka sin flagga från motståndaren. Syftet är att spelarna ska lära sig skydda sin egen flagga och inte bara försöka rusa över och ta motståndarens flagga.  

###### REWARD_PICKUP_FLAG
Spelarna får en reward när de plockar upp motståndarens flagga. Detta är tänkt som ett delmål så att spelarna förstår att de är på rätt väg.

###### REWARD_SCORE_FLAG
En reward som delas ut när spelaren lyckas ta med sig motståndarens flagga tillbaka till sin egen bas. Detta avslutar spelomgången.

###### REWARD_HOMING
Denna hyperparameter ger spelaren en reward när den tar sig närmare motståndarens flagga, men bara när den själv inte bär flaggan. Implementerad för att flagg bäraren snabbar ska nå hem med flaggan.

#### Team Rewards
Team rewards ges ut när en lagspelare utför en uppgift, tanken är att team rewards ska motivera spelarna att samarbete
	REWARD_PICKUP_FLAG_TEAM
	REWARD_SCORE_FLAG_TEAM

#### Grundläggande Penalties

###### PENALTY_HOMING
Ger en penalty när spelaren rör sig bort från motståndarens flagga. Implementerades för att balansera REWARD_HOMING, utan en penalty så kan en agent snabbt lärasig att farma rewrds genom att rörasig fram och tilbaka på spelbrädet. 
    
###### PENALTY_STEP
Denna ger penalty för varje steg, penalty ökar desto längre spelet fortskrider. Tanken är att detta ska vara en *'ticking clock'* som tvinga spelarna att avsluta spelet så fort som möjligt för att undvika automatisk penalty. Man får vara försiktig med denna parameter, sätter man den för högt så ger spelarna upp då inget det gör kan mäta sig med den tickande penalty.

###### PENALTY_COLLISION_PLAYER 
Motverka att spelarna springer in i varandra, vilket kan leda till att de stannar helt. 

###### PENALTY_COLLISION_WALL 
Motverkar att spelarna springer in i väggar, vilket kan leda till att de stannar helt.

###### PENALTY_GETTING_TAGGED 
Tanken är att spelarna ska lära sig att undvika motspelare om de själv har flaggan.

#### Role specific rewards
Tanken är att ge de olika spelarna i laget olika roller, rollerna *'Attacker'* och *'Defender'* delas ut till spelarna vid spelets start, spelaren som startar närmast flaggan blir *Defender*. *Attacker* får extra reward när den rör sig mot fiendens bas, så länge den inte har flaggan och *Defender* får extra rewards när den håller sig närma hemmabasen, defense zone bestämmer hur stort område den defensiva bonusen sträcker sig över.
    
###### REWARD_ATTACKER_PROGRESS 
*Attacker* får en extra belöning för att röra sig mot fiendens flagga.
    
###### REWARD_DEFENDER_RADIUS   & DEFENSE_ZONE
*Defender* får en extra belöning för att vara nära sin egen bas, *defense_zone* avgör hur långt ifrån basen *Defender* kan vara och fortfarande få belöning.
    



## Mitt Bidrag

Alla i projektet va med och utvecklade flera olika delar, vi hade ingen skarp uppdelning utan folk tog över där andra slutade och vi hjälpte varandra. Vissa delar implementerades av en gruppmedlem, andra av två eller tre medlemmar och vissa delar hjälpte alla till med. De följande delar är de som jag spenderade mest tid på.

#### Förundersökning
Det va inte helt självklart att vi skulle köra *Capture the Flag*, vi visste att vi ville testa *Reinforcement Learning* i någon form men inte mer än så. Jag började med att bygga små och väldigt enkla *Reinforcement* project kompletta med neurala nätverk byggda från grunden i torch men jag tog hjälp av LLM till mycket av kodandet.

###### Fire Escape
Mitt första lilla project, designat för att vara så simpelt som möjligt allt agenten behövde lära sig att hitta till utgången i ett litet 6x6 grid. Detta va simpelt att implementera och gick snabbt att träna. Projektet hade potential till att utvecklas till en mer komplex *Maze Solver*, men det va inte riktigt vad gruppen va intresserad av då vi vill ha två agenter som tävlade emot varandra.

![[Fire_Escape.gif]]
###### Hunter & Prey
I detta spel är miljön fortfarande 2d grid men nu har vi istället två agenter med varsitt neuralt nätverk och unika *victory conditions*. Den ena agenten tar rollen som *prey* och likt pack-man får *prey* poäng för varje ny koordinat den besöker. *Prey's victory condition* är att samla ett visst antal poäng.  *Hunter* agentens *victory condition* är att fånga *Prey* agenten.

Detta va ett betydligt mer komplext spel och jag fick implementera betydligt mer rewards för att få agenterna att förstå hur de spelar spelet. Jag implemented *Vision* och gave *Hunter* en större *Vision Box* än *Prey* för att den skulle kunna upptäcka och *Prey* innan *Prey* upptäckte *Hunter*. Eftersom *Hunter* behövde kunna hinna ifatt *Prey* så rör den sig  två steg i taget, vilket gör den snabbare kommer i fatt men får svårare att få tag i *Prey*. Med lite trixande så fick jag agenterna att spela spelet som det va tänkt, *Hunter* jagar *Prey* och ibland hinner *Hunter* i fatt och fångar *Prey* och ibland gör den det inte.

Detta konceptet va mer lovande men vi kände att vi ville ha ett spel där båda agenterna hade samma förutsättning.

![[Hunter_Prey.gif]]
###### Simple Chess 
Här gick jag en liten annan väg och implementerade ett tur baserat brädspel, jag insåg rätt fort att schack va lite väl komplicerat så jag gjorde en mindre version som spelades på ett litet bräde med bara två typer av pjäser, torn och löpare, jag begränsade även deras *movement* till max två rutor per steg. För att göra träningen snabbar satt jag även en *turn limit* och gav vinsten till den spelare som tagit flest av motståndarens pjäser. Detta fungerade bra och agenten hade nu, till skillnad från *Hunter Prey*, sikt över hela brädet och spelade mot sig själv vilket gjorde att samma nätverk och vikter kunde användas för båda sidor. 

Ett problem som snabbt uppstod va att agenten hamnade i ett lokalt minimum då den gjorde exakt samma drag spel efter spel. Detta löstes genom att randomisera startpositionerna på pjäserna och spelet blev betydligt mer dynamiskt. 

![[Simple_Chess.gif]]

###### Capture The Flag
Efter lite diskuterande kom vi fram till att vi vill ha ett *Reinforcement Learning* spel med minst två agenter som hade samma förutsättningar och vi fastnade för *Capture the Flag*. Vi tänkte att detta *CTF* är enkelt nog för att kunna träna relativt snabbt men komplext nog för att vi ska kunna få fram agenterna som visa tecken på lite olika temperament och taktiker. 

Jag slog ihop ett snabbt test med en spelplan, två agenter, väggar och experimenterade lite med lite olika rewards och fick dom att nästan spela spelet. Jag märket att en *heat map* baserad reward va väldigt effektivt, genom att ge agenterna en reward som ökade ju närmare de kom motståndarens flagga så skapade man en kraftfull "*funnel effect*" som snabbt lärde agenterna att snabbt ta sig mot målet.

 Vi gillade konceptet men i slutändan bestämde vi oss för att köra med *MiniGrid*, vi ansåg att de färdigbyggda verktygen skulle göra det snabbare och enklare att träna agenterna.

![[Simple_CTF.gif]]
#### Två mot Två
Det va inte självklart att köra med två spelare i varje lag, en mot en hade troligen varit snabbare och lättare att träna men med två spelare i varje lag öppnade upp möjligheterna för samarbete och mer komplex interaction vilket vägde tyngre än en snabb tränad modell. Jag hade en stor del i implementeringen av två mot två funktionaliteten. 

Allt blev lite mer komplicerat med fyra spelare och de va många små delar i koden som fick skrivas om. 

Det största problemet va att *MiniGrid*  inte va gjort för att hantera två agenter, men min kollega löst detta genom att implementera *PettingZoo* vilket gav oss möjligheten att köra fyra agenter samtidigt. 

Vi va även tvungna att modifiera *MiniGrid* arkitekturen eftersom spelaren aldrig såg sig själv i original versionen, spelaren renderades aldrig i sin egen vision box så vi fick fixa så alla spelare renderades ut så agenten kunde se dom. Färg va också viktigt att ha för nu blev agenterna tvungna att urskilja mellan lagkamrater och motståndare, i en mot en så fans det bara en spelare som agenten kunde se. Jag implementerade även lite visuella ändringar, jag gjorde att spelaren som plockade upp flaggan bytte ikon från en boll till en nyckel, på så sätt kunde agenterna tydligen urskilja om en motståndare bar på flaggan eller inte.

#### Träning och Testning
Under denna del va det verkligen alle man på deck. Projektet lämpade sig väll för grupparbete då alla kunde träna sina egna agenter och mixtra med rewards och hyperparametrar vilket, tycker jag, va den största utmaningen i projekten. Vi kunde sen enkelt testa agenterna genom att köra en tournament vilket va ett roligt och tydligt sätt att utvärdera våra agenter.


## Diskussion
I den här delen så beskriver vi problem som uppstod och hur vi löste dem.

#### Unique Roles
Nackdelen med att basera *CTF* spelet på *Gymnasium* är att all spelare delar på samma vikter vilket kan göra det rörigt för agenten att veta vilken spelare den kontrollerar i stunden. Två sätt att ge agenten mer kontext övervägdes; 

Alternativ 1 är att placera ett object som är unikt för varje spelare i spelarens synfält, detta ger agenten en fast reference som den alltid kan associera med samma spelare. Att byta ut själva spelar objektet hade inte hjälpt eftersom spelaren inte renderas ut och detta visade sig vare en *feature* i *MiniGrid* som va svår att ändra.

Alternativ 2 är att skicka med en binär variable vid sidan av bilden som matchar spelarens klass, *Defender* eller *Attacker*. Fördelen är att det är väldigt smidigt, inget mixtrande med det visuella behövs, nackdelen är att nu skickas både en bild och en variable som bakas ihop i samma Nätverk vilket gör träningen lite mer komplex. 

Trots längre träningstid så valdes alternativ 2 dvs att skicka både en variable och en bild. 

Hade det funnits mer tid så hade de varit intressant att göra en EDA på vikterna och se om de olika spelarna aktiverar olika vikter.

#### Random Number Generator
Det märktes redan vid förundersökningen att slump är en viktig del i tränings processen, en miljö som inte förändras från spel till spel ledder snabbt till att agenten hittade ett mönster av handlingar som den kan implementera varje spelomgång och hamnar på så vis i ett lokalt minimum. Men genom att randomizer startpunkter och sprida ut framslumpade vägg object för varje ny spelomgång så tvingas agenten att anpassa sig sin strategi till miljön och spelet blir betydligt mer dynamiskt och levande.

#### Rewards
För grundläggande träning så räckte det med fyra reward parametrar; plocka upp flaggan, lämna in flaggan, tagga och en penalty för att bli taggad. Detta va allt som behövdes för att få spelarna hitta fram till motståndarens flagg och ta med den tillbaka till sin egen bas. 

Att spelarna medvetet taggade och undvek att bli taggade va dock högst tveksamt alla spelare tenderade till att rusa mot motståndarens flagga och taggsen som skedde va på sin höjd *attack of opportunities*. Så för att försöka få till mer komplexa och taktiska beteenden så introducerades fler reward parametrar. 

Team versioner av de klassiska parametrarna gav spelaren rewards när lagkamraterna gjorde bra i från sig vilket va tänkt att skapa mer taktiska strategier, nackdelen va ett dessa är svåra att utvärdera och troligen hade tagit betydligt längre tid för spelarna att lära sig då de va väldigt avskärmade från eventen som ledde till rewarden.

Ett stort problem som vi stötte på va att agenterna gick in i väggar och vägrade att flytta sig eller att de sprang in i motståndaren och båda vägrade att röra på sig. För att få bukt med dessa introducerades tre nya penalties; en penalty för kollision med väg, en för kollision med spelare och en *ticking clock* penalty som växte för varje steg för att tvinga spelarna att avsluta spelet så fort som möjligt. 

Iden med penalty parametrarna va god men det visade sig vara väldigt känsliga och även små förändringar kunde lätt leda till att dom totalt dominerade och spelarna vägrar att flytta på sig för att undvika minuspoäng vid kollision av vägg eller andra spelare. Men genom att kalkylera den maximal penalty en spelare kunde få under en match så kunde ett mer balanserat värde hittas. 


#### Pick Up Flagg and  Getting Tagged Loop
Det visade sig oerhört viktigt att sätta en relativt hög reward för flagg inlämning annars upptäckte agenten att om avslutade spelet snabbt genom att lämna in flaggan så fick den betydlig mindre rewards än om plockade upp motståndarens flag och lät sig bli taggad om och om igen. Detta ledde till en situation där spelaren, efter att ha plockat upp flaggan, stannade kvar i motståndarens bas.

![[Defensive_Action.gif]]

#### Player Vison
Synfälts aspekten debatterades under projektets gång. Spelet kunde implementeras med full visions för alla spelare men hypotesen va att detta skulle leda till ett mer schack likt beteende, men vad vi va ute efter va att simulera spelare med ett beteende mer likt levande organismer snarare än pjäser på ett bräde. Genom att ge spelarna en ett begränsat synfält så blir de tvungna att söka runt sig i miljön vilket kan sees som ett mer naturligt beteende. 

Ett begränsat synfält gjorde även träning betydligt billigare, hade spelarna haft vision över hela brädet så hade även inputen matchat brädets storlek och då ökat inputen från 84x84x3 (spelarens vision) till 204x204x3 (brädets storlek). En anna fördel med att ha ett begränsat synfält är att inputen till CNN förblir den samma oavsett storleken på spel brädet vilket innebär att `grid_size` nu blir en hyperparameter som kan justeras under tränings processen. 

Genom att börja träna agenten på en liten `grid_size` så lär den sig snabbt att hitta och föra tillbaka flaggan till sin bas. Storleken på miljön kan sedan skalas upp för varje ny tränings omgång. Men det intressanta är att även om agenten tränades på en liten miljö så kom den att prestera lika bra om, inte bättre, på stor miljö i jämförelse med en agent som bara tränats på den stora miljön.

Man kunde tydligt se om en agent tränad på en stor eller liten miljö, agenter tränade på en stor 21x21 miljö använde ofta väggarna längs kanten som guide medans agenter tränade på den mindre miljön 14x14  gick rack över planen, en betydligt kortare väg. 

En annat exempel på guidning kan ses i bilden nedan där golvet på de olika sidorna av miljön har olika färg vilket ger agenten en extra reference punkt som potentiellt kan hjälpa den att hitta fram och tillbaka mellan baserna.

![[Player_Vision.png]]
*Den gula kanten visar vad den röda spelaren, Triangeln, ser*

## Slutsats

**Fråga A:** Hur kan *Reinforcement Learning* implementeras för att träna en agent att spela *Capture the Flag*. 

**Fråga B:** Hur kan hyperparametrar implementeras vid träning för att förbättra agenten dvs få den att spela mer strategiskt och effektivt.

*Reinforcement Learning* kan i högsta grad användas för att lära en agent att spela *Capture the Flag* och även om moduler som *Gymnasium* och *MiniGrid* som använder *Reinforcement Learning* inte är designade för *Multi-Agents Spel* så kan de hackas och kombineras med andra moduler som *PettingZoo* för att skapa ett *CTF* spel med flera agenter.

Att förstå och implementera *Meta Parametrar* som *learning rate*, *batch size*, *frame stacking*, *color channels* osv kan ha en stora konsekvenser för träningstiden och är värt att sätta sig in i. 

*Environment Parametrar* som påverkar spelplanens utformning och svårighetsgrad kan lätt underskattas, att stegvis höja komplexiteten på spelplanen kan ge agenten en snabbare start på träningen. 

Man kan få en agent att spela *CTF* med endast ett fåtal basic *Reward & Penalty Parametrar*, men för mer komplext och strategiskt betande behöver man antingen mer processor kraft och längre träningstid eller introducera fler komplexa *Reward & Penalty Parametrar* dock får man betänka att fler parametrar skapar mer komplicerade interaktioner och där med mer *fine tuning*.


