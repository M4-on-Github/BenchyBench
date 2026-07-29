# Salvage Protocol Reference (condensed, database-grounded)

Source database: `rag_system/Retrieval Databases/new_docs/{aground,capsized,on_fire,sunken}/`.
This is the grounding reference for generating per-image salvage plans in `castor_salvage_plans.csv`.

## Aground

If the vessel is aground, consider whether it can refloat naturally on the next tide
before attempting a forced pull. Larger or deeper-draft vessels usually need
lightering and tug assistance, or dredging if grounded hard. Check all tanks for fuel
or hazardous cargo before any cutting or movement begins.

Sources: ITOPF TIP 18 (Ship Groundings on Coral Reefs), COMDTINST M16130.2F (USCG SAR Addendum),
NOAA Removal of Grounded/Derelict/Abandoned Vessels (Zelo & Helton), A Master's Guide to Shipboard
Accident Response, NOAA PIFSC-156 (Study Design for Vessel Groundings on Coral Reefs).

## Capsized

If the vessel is capsized, prioritize crew rescue before attempting to right the
vessel. Assess overall stability and clear any trapped water before righting, since a
vessel can look stable while still being at risk. Scale the righting method to size —
manual or crane righting for small craft, controlled deballasting or parbuckling for
larger ones — and keep lifting loads controlled to avoid overturning it further.

Sources: Boat Crew Handbook (bch2), Stability Reference Guide (US best practice vessel stability),
IMO Intact Stability Code MSC.267(85).

## Sunken

If the vessel is sunken, test the space for explosive gas, then oxygen, then toxic gas
before any diver enters or hot work begins. Match the recovery method to depth:
patch and refloat in shallow water, patch with a cofferdam and pull in moderate
depths, or crane-and-barge removal in sections if the hull is too deep or too
damaged to raise intact. Offload or account for fuel and cargo before disturbing the
hull, and contain any released oil according to how it behaves in the water.

Sources: US Navy Salvage Manual Vol. 1 (S0300-A6-MAN-010), Salvor's Handbook Rev.2, USS GUARDIAN
Salvage Report, OPRC-HNS/TG Operational Guidelines on Sunken and Submerged Oil, NRT Abandoned
Vessel Authorities and Best Practices Guidance.

## On Fire

If the vessel is on fire, confirm the space is clear of personnel before activating any
fixed suppression system. Match the suppression method to the space — gas
flooding for machinery spaces, foam for cargo decks — and cut off fuel and
ventilation to the fire where possible. Reassess hull and stability once the fire is out,
since fire damage can leave a vessel exposed to sinking or capsizing.

Sources: IMO FSS Code (International Code for Fire Safety Systems), MSC.1/Circ.1321 (fire prevention
in engine rooms/cargo pump rooms), SFLC Std Spec 5550 (2022), MSC.1/Circ.1432.

## Cross-cutting

In every case, let the vessel's size and draft scale the resources needed, and treat
any cargo or hazardous material as reason for added caution rather than a change in
approach.

Available resources may include tugs and beach gear for towing and pulling, cranes
and barges for lifting, pumps and patching materials for dewatering, and
containment booms or recovery systems for spills. Fire response may draw on fire
tugs, foam, and the vessel's own fire pump. Personnel may include a dive and survey
team, a salvage engineer, a firefighting team, and rescue crew, along with gas-testing
equipment before any space is entered. Coordination may involve the vessel owner,
local or flag authorities, a pollution reporting channel, and a commercial salvor.
