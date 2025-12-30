"""
Industrial-Standard Prompt Engineering v3
Enhanced version with comprehensive visual translation, complexity scoring, and expanded concept coverage.

Key Features:
1. Action-to-Visual Translation (dynamic verbs → static descriptions)
2. Composition Enforcement (Rule of Thirds, Center, Spatial)
3. Explicit separation of subjects
4. Negative prompt injections (in positive form)
5. Logic-based template selection
6. Complexity scoring for prompt analysis
7. Expanded concept library (200+ mappings)
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PromptStrategy:
    """Strategy for rendering a specific type of prompt."""
    style: str
    composition: str
    rendering: str
    negative: str

# =============================================================================
# ACTION TO VISUAL TRANSLATION
# Converts dynamic verbs into static positional descriptions
# =============================================================================
ACTION_TO_VISUAL: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Motion / Movement
    # -------------------------------------------------------------------------
    r"\borbit(ing|s)?( around)?\b": "positioned near",
    r"\bcircl(ing|e|es|ed)( around)?\b": "arranged in circular formation around",
    r"\brotat(ing|e|es|ed)\b": "angled view of",
    r"\bspinn(ing|s)?\b": "with radial lines suggesting motion",
    r"\bfly(ing|ies)?\b": "suspended high in empty white space",
    r"\bswooping\b": "diving downward with wings angled",
    r"\bsoar(ing|s)?\b": "gliding with wings fully extended",
    r"\brun(ning|s)?\b": "in dynamic pose with legs spread apart",
    r"\brac(ing|e|es)\b": "in fast motion pose leaning forward",
    r"\bwalk(ing|s)?\b": "in mid-stride stance",
    r"\bjump(ing|s)?\b": "suspended in mid-air above ground",
    r"\bleap(ing|s)?\b": "arcing through the air",
    r"\bswimm(ing|s)?\b": "moving through water with ripple lines",
    r"\bdiv(ing|e|es)\b": "plunging downward head-first",
    r"\bfall(ing|s)?\b": "descending with motion lines",
    r"\bbounc(ing|e|es)\b": "at peak of bounce with spring lines",
    r"\broll(ing|s)?\b": "with circular motion indicators",
    r"\bslid(ing|e|es)\b": "in sliding pose with motion trail",
    r"\bflow(ing|s)?\b": "with curved flowing lines",
    r"\bdanc(ing|e|es)\b": "in expressive pose with arms raised",
    r"\btwirl(ing|s)?\b": "spinning with dress/cape radiating outward",
    
    # -------------------------------------------------------------------------
    # Interactions / Positions
    # -------------------------------------------------------------------------
    r"\bsitt(ing|s)?\b": "resting on top of",
    r"\bstand(ing|s)?\b": "standing upright on",
    r"\blay(ing)?\b|\blying\b": "horizontal and flat on",
    r"\brest(ing|s)?\b": "positioned peacefully on",
    r"\bhold(ing|s)?\b": "with attached",
    r"\bcarry(ing|ies)?\b": "bearing in hands/arms",
    r"\btouch(ing|es)?\b": "in direct contact with",
    r"\bhug(ging|s)?\b": "wrapped around",
    r"\bkiss(ing|es)?\b": "faces close together with",
    r"\bpush(ing|es)?\b": "pressing against",
    r"\bpull(ing|s)?\b": "tugging towards",
    r"\bclimb(ing|s)?\b": "scaling upward on",
    r"\bhang(ing|s)?\b": "suspended from",
    r"\bswing(ing|s)?\b": "arcing suspended from",
    r"\bwav(ing|e|es)\b": "with raised hand gesture",
    r"\bpoint(ing|s)?\b": "with extended finger towards",
    r"\breach(ing|es)?\b": "with arm extended towards",
    r"\bgrab(b?ing|s)?\b": "with hand closed around",
    
    # -------------------------------------------------------------------------
    # Environmental / Atmospheric
    # -------------------------------------------------------------------------
    r"\bshining\b": "with radiating lines emanating outward",
    r"\bglow(ing|s)?\b": "with concentric circles around",
    r"\bsparkl(ing|e|es)\b": "with small star shapes nearby",
    r"\bblaz(ing|e|es)\b": "with flame-like shapes rising",
    r"\bburn(ing|s)?\b": "surrounded by wavy fire lines",
    r"\bsmok(ing|e|es)\b": "with curling smoke wisps above",
    r"\brain(ing|s)?\b": "with vertical lines falling",
    r"\bsnow(ing|s)?\b": "with small dots falling",
    r"\bwind(ing|s|y)?\b": "with horizontal swirl lines",
    r"\bstorm(ing|s|y)?\b": "with zigzag lightning and clouds",
    r"\bfreez(ing|e|es)\b": "with crystalline ice patterns",
    r"\bmelt(ing|s)?\b": "with dripping shapes below",
}

# =============================================================================
# CONCEPT REFINEMENT
# Specific object pairs and concepts that need precise reshaping
# Acts like a "Knowledge Graph" for common pairings
# =============================================================================
CONCEPT_REFINEMENT: Dict[str, str] = {
    # =========================================================================
    # SPACE / CELESTIAL
    # =========================================================================
    r".*moon.*earth.*|.*earth.*moon.*": 
        "a large circle Earth on the left, a small circle Moon on the right, generous negative space between them, both with thick outlines",
    r".*sun.*planet.*|.*planet.*sun.*": 
        "a giant circle Sun with short radiating lines, a tiny circle planet nearby, clear separation",
    r".*solar\s*system.*": 
        "large central circle Sun, smaller circles orbiting at different distances, simple 2D diagram style",
    r".*constellation.*|.*stars.*sky.*": 
        "multiple five-pointed stars scattered across white space, connected by thin lines, night sky pattern",
    r".*galaxy.*|.*spiral.*": 
        "spiral pattern with curved arms radiating from center, simple swirl shape",
    r".*rocket.*|.*spaceship.*": 
        "simple rocket shape: pointed top cone, rectangular body, triangular fins at bottom, flame shape below",
    r".*asteroid.*|.*meteor.*": 
        "irregular jagged rock shape with motion lines trailing behind",
    r".*eclipse.*": 
        "one circle overlapping another, creating partial coverage, high contrast edges",
    
    # =========================================================================
    # ANIMALS - MAMMALS
    # =========================================================================
    r".*cat.*chair.*|.*chair.*cat.*": 
        "a cute cat with pointed triangle ears sitting on top of a simple chair. Cat: round head, oval body, curved tail. Chair: seat, back, four legs. Both clearly visible.",
    r".*cat.*table.*|.*table.*cat.*": 
        "a cat sitting atop a rectangular table. Cat has round head with triangle ears. Table has flat top and four legs.",
    r".*cat.*window.*|.*window.*cat.*": 
        "a cat silhouette in a window frame, cat with triangle ears looking out, rectangular window frame",
    r".*cat.*box.*|.*box.*cat.*": 
        "a cat peeking out of a box, triangle ears visible above box edge, rectangular box shape",
    r".*dog.*bone.*|.*bone.*dog.*": 
        "a dog with floppy ears next to a bone shape, dog sitting or standing, bone with rounded ends",
    r".*dog.*ball.*|.*ball.*dog.*": 
        "a dog in playful pose with a circle ball nearby, dog with tongue out",
    r".*horse.*|.*pony.*": 
        "horse silhouette in profile: long neck, mane lines, four legs, flowing tail",
    r".*elephant.*": 
        "elephant silhouette: large round body, big floppy ears, long trunk curving down, four thick legs",
    r".*giraffe.*": 
        "giraffe silhouette: very long neck, small head, four long legs, short tail, spotted pattern simplified to few spots",
    r".*lion.*": 
        "lion silhouette: round head with mane rays around it, strong body, tail with tuft at end",
    r".*bear.*": 
        "bear silhouette: round thick body, round ears, short legs, sitting or standing pose",
    r".*wolf.*|.*fox.*": 
        "canine silhouette: pointed ears, bushy tail, slender legs, alert standing pose",
    r".*rabbit.*|.*bunny.*": 
        "rabbit with two tall upright ears, round fluffy body, small cotton tail, sitting pose",
    r".*mouse.*|.*rat.*": 
        "small rodent shape: round ears, pointed nose, long thin tail, tiny feet",
    r".*squirrel.*": 
        "squirrel with large bushy curved tail, small round body, holding something in paws",
    
    # =========================================================================
    # ANIMALS - BIRDS
    # =========================================================================
    r".*bird.*tree.*|.*tree.*bird.*": 
        "a bird perching on a tree branch. Bird: round body, pointed beak, wing tucked. Tree: trunk, branches. Both distinct.",
    r".*owl.*": 
        "owl with large round eyes (two circles), pointed ear tufts, round body, sitting on branch",
    r".*eagle.*|.*hawk.*": 
        "bird of prey with wings spread WIDE horizontally, hooked beak, fierce silhouette, majestic soaring pose",
    r".*penguin.*": 
        "penguin silhouette: oval body, tiny wings at sides, pointed beak, standing upright on small feet",
    r".*parrot.*": 
        "colorful bird simplified to silhouette: curved beak, tail feathers, perched on branch",
    r".*swan.*": 
        "elegant swan: long curved S-shaped neck, oval body floating on water line",
    r".*duck.*": 
        "duck silhouette: round body, flat bill, floating on water ripple line",
    r".*flamingo.*": 
        "flamingo: very long thin legs, S-curved neck, round body, standing in water",
    r".*peacock.*": 
        "peacock with large fan tail display: radiating feathers in semicircle pattern behind body",
    
    # =========================================================================
    # ANIMALS - SEA CREATURES
    # =========================================================================
    r".*fish.*": 
        "fish silhouette: oval body, triangular tail fin, dorsal fin on top, simple eye dot",
    r".*shark.*": 
        "shark silhouette: streamlined body, pointed nose, dorsal fin on top, strong tail",
    r".*whale.*": 
        "whale: large streamlined body, small fins, water spout lines above, friendly eye",
    r".*dolphin.*": 
        "dolphin leaping: curved body, pointed snout, dorsal fin, arcing out of water line",
    r".*octopus.*": 
        "octopus: round head, eight curling tentacle lines below, simple eyes",
    r".*jellyfish.*": 
        "jellyfish: dome-shaped head, flowing tentacle lines trailing below",
    r".*crab.*": 
        "crab: wide oval body, two large pincer claws, eight legs spreading outward",
    r".*turtle.*|.*tortoise.*": 
        "turtle: domed shell with pattern, head poking out, four stubby legs, short tail",
    r".*starfish.*|.*sea star.*": 
        "five-pointed starfish shape, thick arms radiating from center",
    
    # =========================================================================
    # ANIMALS - INSECTS & SMALL CREATURES
    # =========================================================================
    r".*butterfly.*": 
        "butterfly: two large wing shapes (upper and lower pairs), thin body in center, antennae on top",
    r".*bee.*": 
        "bee: striped oval body, two small wings, stinger at back, antennae",
    r".*spider.*": 
        "spider: round body, eight long legs radiating outward, maybe on web lines",
    r".*ladybug.*": 
        "ladybug: dome-shaped body with spots, small head, six tiny legs",
    r".*ant.*": 
        "ant: three body segments (head, thorax, abdomen), six legs, antennae",
    r".*dragonfly.*": 
        "dragonfly: long thin body, four elongated wings, large eyes on head",
    r".*snail.*": 
        "snail: spiral shell on back, elongated body below, eye stalks",
    r".*worm.*": 
        "simple curved worm shape, segmented body line",
    
    # =========================================================================
    # PEOPLE / HUMANS
    # =========================================================================
    r".*couple.*|.*lovers.*": 
        "two stick figure people standing side by side. LEFT PERSON: circle head, vertical body, arms, legs. RIGHT PERSON: same. Hands connected by horizontal line. Clear space between except hand connection.",
    r".*people.*holding.*hands.*|.*holding.*hands.*": 
        "two simple stick figures side by side with hands touching in middle. Each: circle head, body, arms, legs. Generous spacing.",
    r".*two.*person.*|.*two.*people.*": 
        "two simple human silhouettes standing apart. Each has distinct head, body, arms, legs.",
    r".*family.*": 
        "group of stick figures: two taller (adults), one or two shorter (children), standing together",
    r".*baby.*|.*infant.*": 
        "small simplified figure: large round head, tiny body, short limbs",
    r".*man.*woman.*|.*woman.*man.*": 
        "two figures side by side: one with short hair, one with long hair or dress triangle, both as simple silhouettes",
    r".*dancer.*": 
        "figure in elegant pose: arms raised, one leg extended, graceful curved lines",
    r".*athlete.*|.*runner.*": 
        "figure in dynamic running pose: leaning forward, arms pumping, legs mid-stride",
    r".*superhero.*": 
        "heroic figure: standing tall, cape flowing, hands on hips, powerful pose",
    
    # =========================================================================
    # ANIMALS - FACE-OFFS / VERSUS
    # =========================================================================
    r".*cat.*vs.*dog.*|.*dog.*vs.*cat.*": 
        "LEFT: cat silhouette with pointed ears, arched back, tail up. RIGHT: dog silhouette with floppy ears, tail wagging. Both facing center. Clear white space between.",
    r".*cat.*and.*dog.*|.*dog.*and.*cat.*": 
        "a cat on left (triangle ears, tail) and a dog on right (floppy ears, snout). Both sitting, clearly separated.",
    
    # =========================================================================
    # GEOMETRIC SHAPES (bold outlines emphasized)
    # =========================================================================
    r".*\bcircle\b.*": 
        "a large bold perfect circle outline, thick black ring, centered, very prominent clean line",
    r".*\btriangle\b.*": 
        "an equilateral triangle pointing UP, sharp apex at top, thick black outline, classic pyramid shape centered",
    r".*\bsquare\b.*": 
        "a large bold square outline, thick black lines, four equal sides, four right angles, centered",
    r".*\brectangle\b.*": 
        "a bold rectangle outline, thick black lines, longer width than height, centered",
    r".*\bpentagon\b.*": 
        "five-sided polygon, regular pentagon shape, thick black outline, centered",
    r".*\bhexagon\b.*": 
        "six-sided polygon, honeycomb cell shape, thick black outline, centered",
    r".*\boctagon\b.*": 
        "eight-sided polygon, stop sign shape, thick black outline, centered",
    r".*\bstar\b.*": 
        "a large bold five-pointed star, thick black outline, classic star shape with five radiating points, centered",
    r".*\bheart\b.*": 
        "a large bold heart shape, classic love heart with two bumps at top and point at bottom, thick outline, centered",
    r".*\bdiamond\b.*": 
        "a rotated square shape (diamond), balanced on one corner, thick black outline, centered",
    r".*\bcross\b.*|.*\bplus\b.*": 
        "a bold cross/plus shape, vertical and horizontal bars of equal width intersecting at center",
    r".*\barrow\b.*": 
        "arrow shape: triangular arrowhead, straight shaft line, simple direction indicator",
    r".*\bspiral\b.*": 
        "spiral curve starting from center and expanding outward, thick line, smooth coil shape",
    r".*\bcube\b.*": 
        "3D cube in isometric view, three visible faces, thick edge lines, geometric solid",
    r".*\bsphere\b.*": 
        "circle with subtle shading lines to suggest 3D roundness, or simple circle with highlight",
    r".*\bcylinder\b.*": 
        "3D cylinder: oval top, rectangular body, oval bottom visible, vertical edge lines",
    r".*\bcone\b.*": 
        "cone shape: circular base, sides tapering to point at top, ice cream cone silhouette",
    r".*\bpyramid\b.*": 
        "pyramid: triangular front face, visible side face, Egyptian pyramid shape",
    
    # =========================================================================
    # ABSTRACT CONCEPTS → CONCRETE VISUALS
    # =========================================================================
    r".*\bhappiness\b.*|.*\bhappy\b.*|.*\bjoy\b.*": 
        "a bright smiling sun face with thick radiating rays, simple circular face with curved smile, two dot eyes, cheerful icon",
    r".*\blove\b.*|.*\bromance\b.*": 
        "a large bold heart shape, thick black outline, classic love heart symbol, centered, iconic",
    r".*\bpeace\b.*": 
        "a peace dove bird silhouette flying with olive branch in beak, simple graceful bird icon",
    r".*\bfreedom\b.*|.*\bliberty\b.*": 
        "a large eagle or bird with wings spread WIDE horizontally, soaring bird silhouette, majestic flying pose",
    r".*\bwisdom\b.*|.*\bknowledge\b.*": 
        "an owl silhouette with large round eyes, or an open book with pages, simple icon",
    r".*\bstrength\b.*|.*\bpower\b.*": 
        "a flexed arm muscle shape, or a lion silhouette, bold strong outline",
    r".*\bhope\b.*": 
        "a sunrise scene: semicircle sun rising above horizon line with rays extending upward",
    r".*\bsadness\b.*|.*\bsad\b.*": 
        "a simple face with downturned mouth, possibly a single tear drop, minimal expression",
    r".*\banger\b.*|.*\bangry\b.*": 
        "a face with furrowed brows (V shape above eyes), downturned mouth, intense expression",
    r".*\bfear\b.*|.*\bscared\b.*": 
        "wide circular eyes, open mouth, hair standing up, scared expression icon",
    r".*\bmusic\b.*": 
        "musical notes: eighth notes, quarter notes, treble clef symbol, floating pattern",
    r".*\btime\b.*": 
        "simple clock face: circle with two hands (short and long) pointing at different angles",
    r".*\bmoney\b.*|.*\bwealth\b.*": 
        "dollar sign symbol, or stack of coins, or piggy bank silhouette",
    r".*\bidea\b.*|.*\bthought\b.*": 
        "light bulb shape: round bulb top, screw base, radiating lines for glow",
    r".*\bdeath\b.*": 
        "simple skull shape, or grim reaper silhouette with scythe, iconic death symbol",
    r".*\blife\b.*": 
        "tree of life: trunk with spreading branches, or sprouting plant, growth symbol",
    r".*\bnature\b.*": 
        "simple tree silhouette, sun, and curved ground line, natural scene elements",
    r".*\btechnology\b.*|.*\btech\b.*": 
        "gear/cog shape, or computer/laptop silhouette, or circuit pattern",
    r".*\bsuccess\b.*|.*\bvictory\b.*": 
        "trophy cup shape, or figure with raised arms in victory pose, or checkered flag",
    
    # =========================================================================
    # BUILDINGS / ARCHITECTURE
    # =========================================================================
    r".*\bhouse\b.*|.*\bhome\b.*": 
        "simple house: triangular roof, rectangular body, centered door, square windows, chimney optional",
    r".*\bcastle\b.*": 
        "castle with crenellated towers, central gate, multiple rectangular towers with flag poles",
    r".*\bchurch\b.*": 
        "church: tall steeple with cross on top, arched door, stained glass windows simplified",
    r".*\bskyscraper\b.*|.*\btower\b.*": 
        "tall rectangular building, many small window squares, antenna on top",
    r".*\blighthouse\b.*": 
        "lighthouse: tall tapered tower, light room at top with radiating beams, coastal setting suggested",
    r".*\bbridge\b.*": 
        "bridge: horizontal span, support pillars below, arch or suspension cables",
    r".*\bwindmill\b.*": 
        "windmill: tall structure with four large rotating blades, X pattern",
    r".*\bpagoda\b.*|.*\btemple\b.*": 
        "Asian pagoda: multi-tiered structure with upturned roof edges at each level",
    r".*\bpyramid.*egypt.*|.*egypt.*pyramid.*": 
        "Egyptian pyramid: triangular face, desert base line, maybe sphinx nearby",
    r".*\beiffel.*tower.*": 
        "Eiffel Tower silhouette: lattice structure widening at base, iconic Paris landmark",
    
    # =========================================================================
    # NATURE / PLANTS
    # =========================================================================
    r".*\btree\b.*": 
        "tree with distinct trunk (rectangle) and leafy crown (cloud/oval shape), grounded on line",
    r".*\bpine.*tree\b.*|.*\bevergreen\b.*|.*\bchristmas.*tree\b.*": 
        "triangular pine tree shape, layered triangles getting smaller toward top, trunk at bottom",
    r".*\bflower\b.*": 
        "flower: circle center, petals radiating around, stem going down, leaves on stem",
    r".*\brose\b.*": 
        "rose: spiral center, layered curved petals, thorny stem, leaves",
    r".*\bsunflower\b.*": 
        "sunflower: large circle center, many elongated petals around edge, thick stem with leaves",
    r".*\btulip\b.*": 
        "tulip: cup-shaped flower head, single stem, two leaves at base",
    r".*\bdaisy\b.*": 
        "daisy: small circle center, many thin petals radiating outward, simple stem",
    r".*\bcactus\b.*": 
        "cactus: tall oval body with arm branches, simple spines as short lines",
    r".*\bpalm.*tree\b.*": 
        "palm tree: tall thin trunk, large fan-like fronds at top radiating outward",
    r".*\bmushroom\b.*": 
        "mushroom: domed cap with spots, short thick stem, simple toadstool shape",
    r".*\bleaf\b.*|.*\bleaves\b.*": 
        "simple leaf shape: oval with pointed tip, center vein line, stem at bottom",
    
    # =========================================================================
    # VEHICLES / TRANSPORTATION
    # =========================================================================
    r".*\bcar\b.*|.*\bautomobile\b.*": 
        "car side view: rectangular body, circular wheels, windows, simple sedan silhouette",
    r".*\btruck\b.*": 
        "truck: cab section, long cargo box behind, large wheels",
    r".*\bbus\b.*": 
        "bus: long rectangular body, many windows, rounded front, large wheels",
    r".*\btrain\b.*|.*\blocomotive\b.*": 
        "train engine: smokestack with puffs, rectangular body, many wheels, cowcatcher",
    r".*\bplane\b.*|.*\bairplane\b.*|.*\baircraft\b.*": 
        "airplane: fuselage cylinder, wing shapes extending from sides, tail fin",
    r".*\bhelicopter\b.*": 
        "helicopter: round cabin, rotor blades on top, tail with small rotor",
    r".*\bboat\b.*|.*\bship\b.*": 
        "boat: hull shape, mast(s) with sails or smokestack, water line below",
    r".*\bsailboat\b.*": 
        "sailboat: triangular sail, curved hull, on water line",
    r".*\bbike\b.*|.*\bbicycle\b.*": 
        "bicycle: two wheels, triangular frame, handlebars, seat, pedals",
    r".*\bmotorcycle\b.*": 
        "motorcycle: two wheels, engine block, seat, handlebars, aggressive pose",
    r".*\bsubmarine\b.*": 
        "submarine: elongated oval body, conning tower on top, propeller at back",
    
    # =========================================================================
    # FOOD / OBJECTS
    # =========================================================================
    r".*\bapple\b.*": 
        "apple: round fruit shape with slight indent at top, small stem, leaf",
    r".*\bpizza\b.*": 
        "pizza: circular pie or triangle slice, toppings as small shapes",
    r".*\bcake\b.*": 
        "cake: rectangular or tiered shape, candles on top, decorative lines",
    r".*\bcupcake\b.*": 
        "cupcake: wrapper at bottom, swirled frosting dome on top, cherry",
    r".*\bcoffee\b.*|.*\btea\b.*": 
        "coffee cup: curved mug handle, steam swirls rising from top",
    r".*\bbook\b.*": 
        "open book: two page rectangles joined at spine, lines for text",
    r".*\bkey\b.*": 
        "key: circular head with hole, rectangular shaft with teeth at end",
    r".*\block\b.*|.*\bpadlock\b.*": 
        "padlock: U-shaped shackle, rectangular body, keyhole",
    r".*\bumbrella\b.*": 
        "umbrella: dome canopy, curved handle at bottom, thin shaft",
    r".*\bballoon\b.*": 
        "balloon: oval/teardrop shape, string hanging below",
    r".*\bgift\b.*|.*\bpresent\b.*": 
        "gift box: cube with ribbon cross on top, bow at center",
    
    # =========================================================================
    # WEATHER / CELESTIAL
    # =========================================================================
    r".*\bsun\b.*": 
        "sun: circle with short radiating lines/rays around it, cheerful solar icon",
    r".*\bmoon\b.*crescent.*|.*crescent.*moon.*": 
        "crescent moon: C-shaped curve, night sky icon",
    r".*\bmoon\b.*": 
        "full moon: circle with subtle crater marks (small circles or dots inside)",
    r".*\bcloud\b.*": 
        "cloud: fluffy shape made of connected circles/bumps, floating in sky",
    r".*\brain\b.*": 
        "rain cloud: cloud shape with vertical lines falling below, droplets",
    r".*\bsnow\b.*": 
        "snowflake: six-pointed crystal pattern, or hexagonal symmetry, delicate lines",
    r".*\blightning\b.*|.*\bthunder\b.*": 
        "lightning bolt: jagged line zigzagging downward, electric energy",
    r".*\brainbow\b.*": 
        "rainbow: arc of curved parallel lines, from horizon to horizon",
    r".*\bsunset\b.*|.*\bsunrise\b.*": 
        "sun semicircle at horizon line, rays extending upward, simple landscape",
    
    # =========================================================================
    # VAGUE / MINIMAL PROMPTS (provide reasonable defaults)
    # =========================================================================
    r"^thing$|^stuff$|^thing\s+on\s+stuff$": 
        "a simple cube shape sitting on a flat surface, geometric objects with thick outlines",
    r"^something$|^anything$|^whatever$": 
        "a random simple shape like a star or circle, bold outline, centered",
    r"^(a\s+)?picture$|^(a\s+)?drawing$|^(a\s+)?image$": 
        "a simple framed scene: rectangle frame, basic landscape inside with sun and tree",
    r"^test$|^hello$|^hi$": 
        "a friendly waving hand icon, simple greeting gesture",
}

# =============================================================================
# STYLE TEMPLATES
# =============================================================================
STYLES = {
    "default": PromptStrategy(
        style="minimalist line art icon, black and white only, pure black lines on PURE WHITE background",
        composition="centered on SOLID WHITE background (#FFFFFF), no texture, no shadows, no gradients, lots of white space around subject",
        rendering="thick bold clean outlines, no shading, no gradients, flat 2D vector style, stark contrast",
        negative="complex, realistic, photo, shading, texture, gray background, colors, messy, sketch lines, blurry, 3d render, textured background, pattern, noise"
    ),
    "diagram": PromptStrategy(
        style="technical diagram style, schematic blueprint view",
        composition="distinct elements with clear separation and labels",
        rendering="high contrast, precise geometric shapes, uniform line weight, engineering style",
        negative="artistic, painterly, textured, messy, overlap, organic"
    ),
    "cartoon": PromptStrategy(
        style="cartoon illustration style, fun and playful",
        composition="dynamic composition with expressive poses",
        rendering="bold black outlines, simplified shapes, exaggerated features, comic style",
        negative="realistic, photograph, muted, serious, complex shading"
    ),
    "icon": PromptStrategy(
        style="app icon style, modern flat design",
        composition="perfectly centered, balanced, symmetrical where appropriate",
        rendering="super thick outlines, maximally simplified, recognizable at any size",
        negative="detailed, complex, thin lines, text, multiple elements"
    ),
}

# =============================================================================
# FEATURE ENHANCEMENT
# Ensures prominent features of objects are explicitly requested
# =============================================================================
FEATURE_ENHANCEMENT: Dict[str, str] = {
    # Animals
    r"\bcat\b": "cat with distinct pointed triangle ears, round eyes, long curved tail",
    r"\bdog\b": "dog with distinct ears, snout with nose, wagging tail",
    r"\bbird\b": "bird with distinct beak, wings, and feathered silhouette",
    r"\bfish\b": "fish with distinct fins, scales pattern, and forked tail",
    r"\bbutterfly\b": "butterfly with symmetrical wing patterns and antennae",
    r"\blion\b": "lion with distinctive mane around face and strong body",
    r"\belephant\b": "elephant with large ears, long trunk, and thick legs",
    r"\bhorse\b": "horse with flowing mane, muscular body, and long legs",
    r"\bsnake\b": "snake with coiled or S-curved body, forked tongue visible",
    r"\bturtle\b": "turtle with patterned shell dome and head poking out",
    
    # Humans / Body
    r"\bface\b": "face with clearly defined eyes, nose, mouth, and expression",
    r"\bperson\b": "person with distinct head, torso, arms, and legs",
    r"\bhand\b": "hand with five fingers clearly separated, palm visible",
    r"\beye\b": "eye with iris circle, pupil, eyelid curve, and lashes",
    
    # Objects
    r"\bhouse\b": "house with distinct triangular roof, door, square windows, chimney",
    r"\btree\b": "tree with distinct vertical trunk and leafy branches/crown",
    r"\bcar\b": "car with distinct windows, two wheels visible, body shape",
    r"\bboat\b": "boat with distinct hull, mast or cabin, on water line",
    r"\bplane\b": "airplane with fuselage, two wings, tail fin clearly visible",
    r"\bflower\b": "flower with distinct petals, center, stem, and leaves",
    r"\bclock\b": "clock with circular face, numbers or marks, two hands",
    r"\bchair\b": "chair with seat, back, and four legs clearly visible",
    r"\btable\b": "table with flat horizontal top surface and four legs",
    r"\bbook\b": "book with rectangular pages, spine, and cover visible",
    
    # Nature
    r"\bmountain\b": "mountain with triangular peak, maybe snow cap at top",
    r"\briver\b": "river with winding curved path, banks on either side",
    r"\bocean\b|sea\b": "ocean with wave patterns, horizontal expanse",
    r"\bcloud\b": "cloud with fluffy rounded bumps, floating shape",
    r"\bstars?\b": "star(s) with five or six pointed shape, radiating",
}

# =============================================================================
# PROMPT ENHANCER CLASS
# =============================================================================
class PromptEnhancer:
    """Enhanced prompt engineering for ASCII art generation."""
    
    def __init__(self):
        self.action_map = ACTION_TO_VISUAL
        self.concept_map = CONCEPT_REFINEMENT
        self.feature_map = FEATURE_ENHANCEMENT
        
    def translate_actions(self, prompt: str) -> str:
        """Translates dynamic actions to static visual descriptions."""
        working_prompt = prompt.lower()
        for pattern, replacement in self.action_map.items():
            if re.search(pattern, working_prompt, re.IGNORECASE):
                working_prompt = re.sub(pattern, replacement, working_prompt, flags=re.IGNORECASE)
        return working_prompt

    def check_concept_override(self, prompt: str) -> Optional[str]:
        """Checks if the concept matches a known complex scenario override."""
        prompt_lower = prompt.lower()
        for pattern, override in self.concept_map.items():
            if re.match(pattern, prompt_lower):
                return override
        return None

    def get_feature_enhancements(self, prompt: str) -> List[str]:
        """Collects specific feature instructions for detected objects."""
        prompt_lower = prompt.lower()
        features = []
        for pattern, enhancement in self.feature_map.items():
            if re.search(pattern, prompt_lower):
                features.append(enhancement)
        return features
    
    def detect_style(self, prompt: str) -> str:
        """
        Detect the best style based on prompt content.
        
        Returns:
            Style key: 'default', 'diagram', 'cartoon', 'icon'
        """
        prompt_lower = prompt.lower()
        
        # Diagram style indicators
        if re.search(r"\b(diagram|schematic|blueprint|technical|anatomy|labeled)\b", prompt_lower):
            return "diagram"
        
        # Cartoon style indicators
        if re.search(r"\b(cartoon|comic|funny|cute|chibi|kawaii|playful)\b", prompt_lower):
            return "cartoon"
        
        # Icon style indicators
        if re.search(r"\b(icon|logo|symbol|app|emoji|minimal|badge)\b", prompt_lower):
            return "icon"
        
        # Check for celestial/scientific content (use diagram)
        if re.search(r"\b(moon|earth|planet|solar|orbit|atom|molecule|cell)\b", prompt_lower):
            return "diagram"
        
        return "default"
    
    def calculate_prompt_complexity(self, prompt: str) -> float:
        """
        Calculate a complexity score for the prompt.
        
        Returns:
            Float from 0.0 (simple) to 1.0 (very complex)
        """
        score = 0.3  # Base score
        prompt_lower = prompt.lower()
        
        # Word count (longer = more complex)
        word_count = len(prompt.split())
        if word_count > 5:
            score += 0.1
        if word_count > 15:
            score += 0.1
        if word_count > 30:
            score += 0.2
        
        # Counting subjects (more subjects = more complex)
        subject_patterns = [
            r"\band\b", r"\bwith\b", r"\bnear\b", r"\bnext to\b",
            r"\babove\b", r"\bbelow\b", r"\bbehind\b", r"\bin front of\b",
        ]
        for pattern in subject_patterns:
            if re.search(pattern, prompt_lower):
                score += 0.05
        
        # Complexity keywords
        complex_words = [
            r"\b(detailed|intricate|complex|realistic|photorealistic)\b",
            r"\b(multiple|several|many|group|crowd)\b",
            r"\b(texture|pattern|gradient|shading)\b",
            r"\b(background|environment|scene|landscape)\b",
        ]
        for pattern in complex_words:
            if re.search(pattern, prompt_lower):
                score += 0.1
        
        # Simplicity keywords (reduce score)
        simple_words = [
            r"\b(simple|minimal|basic|clean|icon)\b",
            r"\b(silhouette|outline|line art|flat)\b",
        ]
        for pattern in simple_words:
            if re.search(pattern, prompt_lower):
                score -= 0.1
        
        return max(0.0, min(1.0, score))

    def enhance(self, prompt: str, style_override: Optional[str] = None) -> str:
        """
        Main entry point for prompt enhancement.
        
        Args:
            prompt: Original user prompt
            style_override: Force a specific style ('default', 'diagram', 'cartoon', 'icon')
            
        Returns:
            Enhanced prompt optimized for ASCII art generation
        """
        
        # 1. Check for full concept override (highest accuracy for specific known hard cases)
        override = self.check_concept_override(prompt)
        if override:
            core_prompt = override
            style_key = style_override or self.detect_style(prompt)
            strategy = STYLES.get(style_key, STYLES["default"])
            
            # Append features even to overrides if applicable
            additional_features = self.get_feature_enhancements(prompt)
            if additional_features:
                # Only add features not already in the override
                for feature in additional_features:
                    if feature.split()[0] not in core_prompt:
                        core_prompt += f", {feature}"
        else:
            # 2. General Translation
            core_prompt = self.translate_actions(prompt)
            style_key = style_override or self.detect_style(prompt)
            strategy = STYLES.get(style_key, STYLES["default"])
            
            # Add specific feature enhancements
            additional_features = self.get_feature_enhancements(prompt)
            if additional_features:
                core_prompt += ", " + ", ".join(additional_features)

        # 3. Construct Final Prompt
        # Formula: [Style] + [Composition] + [Subject/Core] + [Visual Enforcers]
        
        visual_enforcers = (
            "distinct silhouettes, generous white space between elements, "
            "clean sharp edges, prominent features clearly visible, "
            "simple geometric primitives"
        )
        
        final_prompt = (
            f"{strategy.style}, {strategy.composition}, "
            f"SUBJECT: {core_prompt}, "
            f"{strategy.rendering}, "
            f"{visual_enforcers}"
        )
        
        return final_prompt
    
    def get_negative_prompt(self, prompt: str) -> str:
        """
        Get negative prompt based on detected style.
        
        Returns:
            Negative prompt string for Stable Diffusion
        """
        style_key = self.detect_style(prompt)
        strategy = STYLES.get(style_key, STYLES["default"])
        return strategy.negative


# =============================================================================
# SINGLETON & PUBLIC API
# =============================================================================
enhancer = PromptEnhancer()


def enhance_prompt(prompt: str, style: Optional[str] = None) -> str:
    """
    Public API for the enhancer.
    
    Args:
        prompt: User's original prompt
        style: Optional style override ('default', 'diagram', 'cartoon', 'icon')
        
    Returns:
        Enhanced prompt optimized for ASCII art
    """
    return enhancer.enhance(prompt, style_override=style)


def get_negative_prompt(prompt: str) -> str:
    """Get negative prompt for the given input."""
    return enhancer.get_negative_prompt(prompt)


def get_complexity_score(prompt: str) -> float:
    """Get complexity score for a prompt (0.0 to 1.0)."""
    return enhancer.calculate_prompt_complexity(prompt)


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PROMPT ENGINEERING V3 - TEST SUITE")
    print("=" * 70)
    
    tests = [
        # Abstract concepts
        "freedom",
        "happiness",
        "love",
        
        # Simple subjects
        "a cat",
        "a simple house",
        "a tree",
        
        # Relationships
        "moon orbiting earth",
        "a cat sitting on a chair",
        "a bird flying",
        "cat and dog together",
        
        # Complex (should be simplified)
        "a beautiful detailed realistic forest with many animals",
        "photorealistic 3D cat with fur texture",
        
        # Geometric
        "a triangle",
        "a star",
        "a heart",
        
        # Vague
        "thing on stuff",
        "something",
    ]
    
    for t in tests:
        complexity = get_complexity_score(t)
        enhanced = enhance_prompt(t)
        negative = get_negative_prompt(t)
        
        print(f"\n📝 IN:  \"{t}\"")
        print(f"   Complexity: {complexity:.2f}")
        print(f"✨ OUT: {enhanced[:100]}...")
        print(f"🚫 NEG: {negative[:50]}...")
        print("-" * 40)
