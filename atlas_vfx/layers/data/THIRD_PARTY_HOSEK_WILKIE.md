# Hosek-Wilkie sky model data

`hosek_wilkie_spectral.npz` holds the coefficients of the Hosek-Wilkie analytic skylight model (2012) and
its solar radiance function (2013), version 1.4a, converted unchanged from the authors' sample
implementation as carried by pbrt-v4 at commit 9772673e43e74b931675081b2826f4acc91ba348 (`src/ext/skymodel`). The file hashes are in
`hosek_wilkie_spectral.json`. `atlas_vfx/layers/sky_hosek.py` is a numpy port of the same formulas.

L. Hosek, A. Wilkie, "An Analytic Model for Full Spectral Sky-Dome Radiance", SIGGRAPH 2012.
L. Hosek, A. Wilkie, "Adding a Solar Radiance Function to the Hosek Skylight Model", IEEE CG&A 2013.

The source's licence, verbatim:

```
This source is published under the following 3-clause BSD license.

Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote 
      products derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
