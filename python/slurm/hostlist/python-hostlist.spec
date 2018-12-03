%{!?python_sitelib: %define python_sitelib %(%{__python} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()")}

Name:           python-hostlist
Version:        1.18
Release:        1
Summary:        Python module for hostlist handling
Vendor:         NSC

Group:          Development/Languages
License:        GPL2+
URL:            http://www.nsc.liu.se/~kent/python-hostlist/
Source0:        http://www.nsc.liu.se/~kent/python-hostlist/%{name}-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

BuildArch:      noarch
BuildRequires:  python-devel

%description
The hostlist.py module knows how to expand and collect hostlist
expressions. The package also includes the 'hostlist' binary which can
be used to collect/expand hostlists and perform set operations on
them.

%prep
%setup -q


%build
%{__python} setup.py build --executable="/usr/bin/python -E"


%install
rm -rf $RPM_BUILD_ROOT
%{__python} setup.py install -O1 --skip-build --prefix /usr --root $RPM_BUILD_ROOT

 
%clean
rm -rf $RPM_BUILD_ROOT


%files
%defattr(-,root,root,-)
%doc README
%doc COPYING
%doc CHANGES
%{python_sitelib}/*
/usr/bin/hostlist
/usr/bin/hostgrep
/usr/bin/pshbak
/usr/bin/dbuck
/usr/share/man/man1/hostlist.1.gz
/usr/share/man/man1/hostgrep.1.gz
/usr/share/man/man1/pshbak.1.gz
/usr/share/man/man1/dbuck.1.gz
%changelog
* Thu Jul 21 2018 Kent Engström <kent@nsc.liu.se> - 1.18-1
- Accept whitespace in hostlists passed as arguments
- Support both Python 2 and Python 3 natively

* Mon Jan 23 2017 Kent Engström <kent@nsc.liu.se> - 1.17-1
- New features in dbuck by cap@nsc.liu.se:
- Add option -z, --zero
- Add option -b, --bars
- Add option --highligh-hostlist and --color
- Add option -a, --anonymous
- Add option -p, --previous and --no-cache
- Also other fixes and cleanups in dbuck

* Mon May 23 2016 Kent Engström <kent@nsc.liu.se> - 1.16-1
- Ignore PYTHONPATH et al. in installed scripts

* Thu Apr 21 2016 Kent Engström <kent@nsc.liu.se> - 1.15-1
- Add missing options to the hostgrep(1) man page.
- Add --restrict option to hostgrep.
- Add --repeat-slurm-tasks option.
- dbuck: major rewrite, add -r/-o, remove -b/-m
- dbuck: add a check for sufficient input when not using -k
- dbuck: Fix incorrect upper bound of underflow bucket
