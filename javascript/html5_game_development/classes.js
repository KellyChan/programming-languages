

// Create an inheritance tree using Class.extend() of the
// following form:
//
// 1) Weapon should extend Class.
//
// 2) MachineGun should extend Weapon.
//
// 3) ChainGun should extend Weapon.
//
// 4) Entity should extend Class.
//
// 5) Teleporter should extend Entity.
//
// 6) EnergyCanister should extend Entity.
//
// We've started things off for you by doing steps
// (1) and (2).

Weapon = Class.extend({
    init: function() {
        
    }
});

MachineGun = Weapon.extend({
    init: function() {
        
    }
});

// YOUR CODE HERE
Weapon = Class.extend(); 
MachineGun = Weapon.extend();
ChainGun = Weapon.extend();
Entity = Class.extend();
Teleporter = Entity.extend();
EnergyCanister = Entity.extend();







