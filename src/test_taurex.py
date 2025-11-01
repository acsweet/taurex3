import numpy as np
import matplotlib.pyplot as plt
from taurex.parameter import ParameterParser
from taurex.output.hdf5 import HDF5Output

# Path to your parameter file
parameter_file = '../test_files/quickstart.par'

# Initialize parser and read configuration
pp = ParameterParser()
pp.read(parameter_file)

# Set up global parameters
pp.setup_globals()

# Generate observation (if any)
observation = pp.generate_observation()
print('observation', type(observation))

# Set up binning
binning = pp.generate_binning()
print('binning', type(binning))

# Generate appropriate model
model = pp.generate_appropriate_model(obs=observation)

# Build the model
model.build()

wngrid = None
if binning is None:
    if observation is None or observation == "self":
        binning = model.defaultBinner()
        wngrid = model.nativeWavenumberGrid
    else:
        binning = observation.create_binner()
        wngrid = observation.wavenumberGrid
else:
    if binning == "native":
        binning = model.defaultBinner()
        wngrid = model.nativeWavenumberGrid
    elif binning == "observed":
        binning = observation.create_binner()
        wngrid = observation.wavenumberGrid
    else:
        binning, wngrid = binning

# Generate instrument (if any)
instrument = pp.generate_instrument(binner=binning)
num_obs = 1
if instrument is not None:
    instrument, num_obs = instrument

instrument_result = None
if instrument is not None:
    instrument_result = instrument.model_noise(
        model, model_res=model.model(), num_observations=num_obs
    )

optimizer = None
solution = None

# Run forward model
result = model.model()

# Optionally run retrieval
run_retrieval = False  # Set to True if you want to run a retrieval
if run_retrieval and observation is not None:
    optimizer = pp.generate_optimizer()
    optimizer.set_model(model)
    optimizer.set_observed(observation)
    pp.setup_optimizer(optimizer)
    
    solution = optimizer.fit()
    
    # Update model with best fit parameters
    for _, optimized, _, _ in optimizer.get_solution():
        optimizer.update_model(optimized)
        break
    
    # Re-run model with optimized parameters
    result = model.model()

# Save output to HDF5
output_file = 'output.hdf5'
with HDF5Output(output_file) as o:
    model.write(o)
    
    out = o.create_group("Output")
    if observation is not None:
        obs = o.create_group("Observed")
        observation.write(obs)
    
    profiles = model.generate_profiles()
    spectrum = binning.generate_spectrum_output(result)
    
    if instrument_result is not None:
        spectrum["instrument_wngrid"] = instrument_result[0]
        spectrum["instrument_wnwidth"] = instrument_result[-1]
        spectrum["instrument_wlgrid"] = 10000 / instrument_result[0]
        spectrum["instrument_spectrum"] = instrument_result[1]
        spectrum["instrument_noise"] = instrument_result[2]
    
    # Store contributions if needed
    try:
        from taurex.util.output import store_contributions
        spectrum["Contributions"] = store_contributions(binning, model)
    except Exception:
        pass
    
    out.store_dictionary(profiles, group_name="Profiles")
    out.store_dictionary(spectrum, group_name="Spectra")


if run_retrieval and solution:
    # Plot the posteriors and best-fit spectrum
    
    # 1. For the best-fit spectrum (already included in your code)
    # The 'result' variable already contains the model with optimized parameters
    
    # 2. For the posteriors
    if optimizer:
        import corner
        
        # Extract samples and parameter names from the optimizer
        samples = optimizer.samples
        params = optimizer.fit_names
        
        # Create the corner plot (posterior distributions)
        fig_corner = corner.corner(
            samples,
            labels=params,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
        )
        
        plt.figure(fig_corner.number)
        plt.savefig('posteriors.png')
        plt.close(fig_corner)

# Plotting
wlgrid = 10000 / wngrid
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"$(R_p/R_s)^2$")
ax.set_xscale("log")

if observation is not None:
    ax.errorbar(
        observation.wavelengthGrid,
        observation.spectrum,
        observation.errorBar,
        fmt=".",
        label="observation",
    )

if instrument_result is not None:
    from taurex.util import wnwidth_to_wlwidth
    
    inst_wngrid, inst_spectrum, inst_noise, inst_width = instrument_result
    inst_wlgrid = 10000 / inst_wngrid
    inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)
    
    ax.errorbar(
        inst_wlgrid,
        inst_spectrum,
        inst_noise,
        inst_wlwidth / 2,
        ".",
        label="Instrument",
    )
else:
    ax.plot(wlgrid, binning.bin_model(result)[1], label="forward model")

# Plotting contributions
show_contrib = False  # Set to True to show basic contributions
if show_contrib:
    native_grid, contrib_result = model.model_contrib(wngrid=wngrid)
    
    for contrib_name, contrib in contrib_result.items():
        flux, tau, extras = contrib
        
        binned = binning.bindown(native_grid, flux)
        ax.plot(wlgrid, binned[1], label=contrib_name)

plt.legend()
plt.show()