module app;

import nloptd.raw : nlopt_version;

void main()
{
    int major, minor, patch;
    nlopt_version(&major, &minor, &patch);
    assert(major >= 2);
}
