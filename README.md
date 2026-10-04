This repository contains the source code, training scripts, and evaluation scripts developed for my MSc Individual Research Project (IRP) at Cranfield University, submitted as part of the requirements for the degree of MSc Autonomous Vehicle Dynamics and Control.

Author: Trisha Neelakantam (485083)

Supervisors: Dr Sabyasachi Mondal and Prof. Ivan Petrunin

Institution: Centre for Autonomous and Cyber-Physical Systems, School of Aerospace, Transport and Manufacturing, Cranfield University

Project Overview:

Accurate state and velocity estimation remains a fundamental challenge in autonomous navigation, particularly in GNSS-denied environments where dead reckoning relies exclusively on onboard inertial measurement units (IMUs). While traditional double-integration causes sensor errors to compound exponentially into severe trajectory drift, unconstrained deep learning models frequently violate physical laws during aggressive manoeuvres.

This project develops a Physics-Informed Neural Network integrated with a Gated Recurrent Unit (PINN-GRU) architecture. By embedding multi-objective physical constraints directly into the network's training loop—specifically baseline kinematic MSE, heading alignment, kinetic energy preservation, and lateral cross-product constraints—the framework successfully restricts cumulative integration drift without requiring exhaustive platform calibration.
