import numpy as np
from sgp4.api import Satrec, jday
from twob import two_body_analytical_update 

# Gravitational constant in km^3 kg^-1 s^-2
G = 6.67430e-20

def validate_analytical_update(tle_line1, tle_line2, dt):
    """
    Validate the two_body_analytical_update function by comparing its output with SGP4 propagation.
    Positions in km, velocities in km/s.
    """
    # Initialize SGP4 satellite
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # Initial epoch from TLE
    # Extract epoch from TLE for accurate propagation
    epoch_year = int(tle_line1[18:20])
    epoch_day = float(tle_line1[20:32])
    
    # Convert TLE epoch to Julian date

    # Assuming epoch_year is 2025
    if epoch_year < 57:
        epoch_year += 2000
    else:
        epoch_year += 1900
    
    # SGP4 uses the epoch as the initial time
    jd_epoch, fr_epoch = jday_from_epoch(epoch_year, epoch_day)
    
    # Propagate to the initial epoch to get position and velocity
    e0, r0, v0 = satellite.sgp4(jd_epoch, fr_epoch)
    if e0 != 0:
        raise RuntimeError(f"SGP4 initial propagation error, code {e0}")
    
    # Initialize positions and velocities
    pos1 = np.array([0.0, 0.0, 0.0])  # Earth at origin in inertial frame
    vel1 = np.array([0.0, 0.0, 0.0])  # Earth stationary
    
    pos2 = np.array(r0)  # Satellite position in km
    vel2 = np.array(v0)  # Satellite velocity in km/s
    
    mass_earth = 5.972e24  # Earth's mass in kg
    mass_satellite = 500.0  # Satellite mass in kg (arbitrary, since two-body is mass-independent for the satellite)
    
    # Run the twobody analytical update function
    pos1_next, vel1_next, pos2_next, vel2_next = two_body_analytical_update(
        pos1, vel1, mass_earth,
        pos2, vel2, mass_satellite,
        dt
    )
    
    # Propagate SGP4 to t + dt
    # SGP4 expects time in days; convert dt from seconds to days
    dt_days = dt / 86400.0
    jd_next = jd_epoch + (fr_epoch + dt_days)
    e1, r1, v1 = satellite.sgp4(jd_next, 0.0)  # fr is already accounted for in dt_days
    if e1 != 0:
        raise RuntimeError(f"SGP4 propagation error at t+dt, code {e1}")
    
    real_pos = np.array(r1)  # Satellite position in km at t + dt
    real_vel = np.array(v1)  # Satellite velocity in km/s at t + dt
    
    # Compute differences
    pos_diff = np.linalg.norm(real_pos - pos2_next)  # km
    vel_diff = np.linalg.norm(real_vel - vel2_next)  # km/s
    
    # Define tolerances
    pos_tolerance = 1.0  # Acceptable difference in km
    vel_tolerance = 0.1  # Acceptable difference in km/s
    
    # Check if the function passes the test
    position_check = pos_diff <= pos_tolerance
    velocity_check = vel_diff <= vel_tolerance
    
    # Report results
    print(f"Position difference: {pos_diff:.3f} km")
    print(f"Velocity difference: {vel_diff:.3f} km/s")
    print("Validation result:")
    if position_check and velocity_check:
        print("✅ The two_body_analytical_update function matches the real data within tolerance.")
    else:
        print("❌ The two_body_analytical_update function does not match the real data within tolerance.")
    return position_check and velocity_check

def jday_from_epoch(year, day):
    """
    Convert a year and day of year to Julian date.
    """
    from datetime import datetime, timedelta

    # Create datetime object for January 1st of the given year
    dt = datetime(year, 1, 1) + timedelta(days=day - 1)
    
    # Convert to Julian date (Algorithm from https://en.wikipedia.org/wiki/Julian_day#Calculation)
    a = (14 - dt.month)//12
    y = dt.year + 4800 - a
    m = dt.month + 12*a - 3
    jd = dt.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045
    # Fractional day
    frac_day = dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
    return jd, frac_day

if __name__ == "__main__":
    # Sample TLE (From test_tle.txt: CALSPHERE 1)
    tle_line1 = "1 00900U 64063C   25009.89296725  .00000966  00000+0  98808-3 0  9998"
    tle_line2 = "2 00900  90.2090  59.8649 0027333  79.0263 331.4307 13.75736827999442"

    dt = 100.0  # Time step in seconds
    
    # Validate the function
    try:
        validation_passed = validate_analytical_update(
            tle_line1, tle_line2, dt
        )
    except RuntimeError as e:
        print(f"Validation failed: {e}")