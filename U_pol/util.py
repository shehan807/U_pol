#!/usr/bin/env python
def get_raw_inputs(simmd, system, nonbonded_force, drude_force):
    positions = simmd.context.getState(getPositions=True).getPositions()
    r = []
    q = []
    Drude = []
    
    # Loop over regular particles
    for i in range(system.getNumParticles()):
        # Get charge, sigma, epsilon for each atom
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        charge = charge.value_in_unit(elementary_charge)

        # Get position of the atom
        pos = list(positions[i])
        pos = [p.value_in_unit(nanometer) for p in pos]

        # Check if this atom has an associated Drude particle
        has_drude = False
        for j in range(drude_force.getNumParticles()):
            # Retrieve Drude particle parameters
            params = drude_force.getParticleParameters(j)
            parent_atom_index = params[0]
            
            polarizability = params[6]
            if parent_atom_index == i:  # If Drude particle is associated with this atom
                has_drude = True
                Drude.append(True)
        
        if not has_drude:
            Drude.append(False)
        
        q.append(charge)
        r.append(pos)
        # Output relevant information
    
    logger.debug("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    logger.debug("\nq")
    logger.debug(q)
    logger.debug("\nr")
    logger.debug(r)
    logger.debug("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

def get_DrudeTypeMap(forcefield):
    """
    For a given openmm force field, get the 'DrudeTypeMap'
    to distinguish the name of Drude atom types. 
    """
    from openmm.app.forcefield import DrudeGenerator
    drudeTypeMap = {}
    for force in forcefield._forces:
        if isinstance(force, DrudeGenerator):
            for type in force.typeMap:
                drudeTypeMap[type] = force.typeMap[type][0]

