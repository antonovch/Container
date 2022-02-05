classdef Container
    properties (Hidden = false)
        data
        params struct
        opts
    end
    
    methods
        function obj = Container(data, varargin)
            if length(varargin) > 1
                params = struct(varargin{:});
            else
                params = varargin{1};
            end
            %%% some checks
            if numel(params) > 1, params = squeezestruct(params); end
            lens = cellfun(@length,struct2cell(params));
            sz = size(data);
            test = all(lens(1:numel(sz)) == sz(:)) && all(lens(numel(sz)+1:end) == 1);
            assert(test, ['Number of parameters does ', ...
            'not correspond to the shape of the data tensor.'])
        
            obj.data = data;
            obj.params = params;
        end
        
        function varargout = subsref(self, s)
            if strcmp(s(1).type, '.')
                % special case: if dot-refencing params field, substitute
                % container self with self.params struct and proceed
                % normally with the builtin subsref 
                if isfield(self(1).params, s(1).subs)
                    self = reshape([self.params], builtin('size',self));
                elseif isfield(self(1).opts, s(1).subs)
                    self = reshape([self.opts], builtin('size',self));
                elseif ~any(strcmp(properties('Container'), s(1).subs))
                    % this allows to run functions/methods with not
                    % notation. Methods were supported all along, of course
                    self = feval(s(1).subs, self);
                    s = s(2:end);
                end
            elseif numel(self) == 1
                switch s(1).type
                    case '()'
                        if ischar(s(1).subs{1}) && ~strcmp(s(1).subs{1}, ':')
                            subs = repmat({':'}, 1, numel(size(self.data)));
                            pnames = fieldnames(self.params);
                            fnames = s(1).subs(1:2:end);
                            vals = s(1).subs(2:2:end);
                            for ii = 1:numel(fnames)
                                %%% convert cell array of chars to array of
                                %%% stings
                                if iscell(vals{ii}) && ischar(vals{ii}{1})
                                    vals{ii} = cellfun(@string, vals{ii});
                                end
                                % insert in the position that corresponds to
                                % the given parameter. We use find for
                                % simplicity, since there should be only a
                                % single match.
                                % subs{isub} will contain a logical array of
                                % length(self.params.(pnames{isub})) that will
                                % tell which values of data to return
                                isub = find(strcmpi(pnames, fnames{ii}));
                                if isnumeric(vals{ii})
                                    parvals = self.params.(pnames{isub}); % allows to igonre case
                                    cmp = parvals(:) == vals{ii};
                                else % string (can be an array)
                                    [M1, M2] = ndgrid(self.params.(pnames{isub}), string(vals{ii}));
                                    cmp = strcmpi(M1,M2);
                                end
                                check = any(cmp,1);
                                if ~all(check)
                                    f = num2cell(find(~check));
                                    str = repmat('%.0f, ', 1, length(f));
                                    str = str(1:end-2);
                                    if length(f) > 1
                                        plural = 's'; 
                                        str = ['[',str,']']; %#ok
                                    else
                                        plural = ''; 
                                    end
                                    str = sprintf(str, f{:});
                                    error('Provided value%s #%s for parameter %s not found.', plural,str,pnames{isub})
                                end
                                self.params.(pnames{isub}) = vals{ii};
                                subs{isub} = any(cmp, 2);
                            end
                            self.data = self.data(subs{:});
                            s = s(2:end);
                        else % numerical, the usual indexing
                            % fill the remainding fields with ':'
                            s(1).subs = [s(1).subs, repmat({':'}, 1, numel(size(self.data))-numel(s(1).subs))];
                            pnames = fieldnames(self.params);
                            for ii = 1:length(s(1).subs)
                                if ~strcmp(s(1).subs{ii}, ':')
                                    self.params.(pnames{ii}) = subsref(self.params.(pnames{ii}), struct('type','()','subs',{s(1).subs(ii)}));
                                end
                            end
                            self.data = self.data(s(1).subs{:});
                            s = s(2:end);
                        end
                    case '{}'
                        if iscell(self.data)
                            varargout = subsref(self, struct('type','()','subs',{s(1).subs}));
                            if numel(s) > 1
                                if numel(varargout) == 1
                                    varargout = {builtin('subsref', varargout{1}, s(2:end))};
                                else
                                    error(['Expected one output from a curly brace ',...
                                        'or dot indexing expression, but there were 2 results.']);
                                end
                            end
                            return
                        else
                            error('Brace indexing is not supported for data of class %s.', class(self.data))
                        end
                end
            end
            if isempty(s)
                varargout = {self};
            else
                varargout = cell(1,max(1, nargout));
                [varargout{:}] = builtin('subsref', self, s);
%                 varargout = builtin('subsref', self, s);
%                 if ~iscell(varargout)
%                     varargout = {varargout};
%                 end
            end
            varargout(cellfun(@isempty, varargout)) = [];
        end
        
        function self = squeeze(self)
        % We do not use deafult squeeze here, because it leaves 1xN vector
        % same size, but we want to have Nx1
            lens = cellfun(@length,struct2cell(self.params)); % get the lengths of the parameters
            fnames = fieldnames(self.params);
            fnames = fnames(lens == 1);
            self.params = rmfield(self.params, fnames);
            lens = lens(lens ~= 1);
            if numel(lens) == 1, lens = [lens 1]; end
            self.data = reshape(self.data, lens(:).');
        end
        
        function disp(self)
            if builtin('numel',self) > 1
                builtin('disp',self); 
                return
            end
            fnames = fieldnames(self.params);
            maxlength = max(cellfun(@length, fnames));
            fprintf('<a href="matlab:helpPopup Container">Container</a> with data tensor of rank %g.\n', numel(fnames))
            fprintf('The parameters are:\n')
            for ii = 1:length(fnames)
                fname = fnames{ii};
                fprintf('\t%*s: ', maxlength, fname);
                val = self.params.(fname);
                switch class(val)
                    case 'double'
                        if length(val) < 6
                            fprintf([num2str(val),'\n'])
                        elseif length(unique(diff(val))) == 1
                            fprintf('%g : %g : %g\n', val(1), diff(val(1:2)), val(end))
                        else
                            str = repmat('%gx',1,length(size(val)));
                            str = [str(1:end-1), ' %s'];
                            valstr = sprintf(str, size(val), class(val));
                            fprintf('%s (%.1f to %.1f)\n', valstr, val(1), val(end));
                        end
                    case {'string','char'}
                        if ischar(val), val = string(val); end
                        str = arrayfun(@(n) [num2str(n),':', char(val(n)), '  '],1:numel(val),'UniformOutput',false);
                        fprintf([[str{:}],'\n'])
                    case 'cell'
                        fprintf('%gx%g cell array\n', size(val));
                end
            end
        end
        
        function tf = eq(A,B)
        % tf = eq(A,B)
        % Reloaded '==' operator. Data property is not compared, because
        % checking params is enough to compare its size up tp a permutation.
        % This asnwers the question whether the data was calculated for the
        % same set of sweep parameters. But the data itself can be
        % different due to simulation details. Use isequal to compare all
        % the properties.
            fA = cellfun(@lower, fieldnames(A.params), 'UniformOutput', false);
            fB = cellfun(@lower, fieldnames(B.params), 'UniformOutput', false);
            sA = cell2struct(struct2cell(A.params), fA, 1);
            sB = cell2struct(struct2cell(B.params), fB, 1);
            tf = isequal(sA, sB); % case-sensitive, but not order
        end
        
        function A = order(A,B)
            [A.params, I] = orderfields(A.params, B.params);
            A.data = permute(A.data, I);
        end
        
        function c = rename(c, param, name)
            fnames = fieldnames(c.params);
            c.params.(name) = c.params.(param);
            c.params = rmfield(c.params, param);
            fnames(strcmp(fnames,param)) = name;
            c.params = orderfields(c.params,fnames);
        end
        
        function varargout = size(c, varargin)
        % for an array of Containers, returns the size of the array. Could
        % make it to return size of the data tensor of each Container.
        % Developer's choice.
            varargout = cell(1,max(1,nargout));
            if numel(c) > 1
                [varargout{:}] = builtin('size', c, varargin{:});
            else
                [varargout{:}] = size(c.data, varargin{:});
            end
        end
        
        function M = full(self)
            M = full(self.data);
        end
        
        %%% messes up subsref when asking for a property
%         function N = numel(c, varargin)
%             if isempty(varargin)
%                 N = numel(c.data);
%             else
%                  N = numel(subsref(c,struct('type','()','subs',{varargin})));
%             end
%         end

        function A = horzcat(A,varargin)
%             A = cat(1, A, varargin{:});
            A = builtin('horzcat',A,varargin{:});
        end
        
        function A = vertcat(A, varargin)
%             A = cat(2, A, varargin{:});
            A = builtin('vertcat',A,varargin{:});
        end
        
        function A = cat(dim,A,varargin)
            % quick and dirty support for arbitrary number of Containers
            switch length(varargin)
                case 1
                    B = varargin{1};
                case 0
                    return
                otherwise
                    for ii = 1:length(varargin)
                        A = cat(dim,A,varargin{ii});
                    end
                    return                
            end
            fnames = fieldnames(A.params);
            if ischar(dim)
                dim = find(strcmpi(fnames, dim));
            end
            A = order(A,B); % orders and checks that the fields are equal
            valA = struct2cell(A.params);
            valAdim = valA{dim};
            valA(dim) = [];
            valB = struct2cell(B.params);
            valBdim = valB{dim};
            valB(dim) = [];
            assert(isequal(valA,valB),'All parameters besides #%g have to be equal',dim)
            % doesn't work with function_handles or cells
            test = @(x) ~iscell(x) || (iscell(x) && ~isa(x{1}, 'function_handle') && ~iscell(x{1}));
            if test(valAdim) && test(valBdim)
                assert(numel(intersect(valAdim,valBdim))==0, ...
                    'Common elements along dimension %g are present.', dim)
            end
            A.params.(fnames{dim}) = [A.params.(fnames{dim}) B.params.(fnames{dim})];
            A.data = cat(dim, A.data, B.data);
        end
        
        function n = nnz(c)
            n = nnz(c.data);
        end
        
        function c = merge(c,varargin)
            I = find(cellfun(@ischar, varargin));
            parm = varargin{I};
            val = varargin{I+1}(:).';
            varargin(I:I+1) = [];
            cc = [varargin{:}];
            if iscell(c), cc = [cc{:}]; end
            if numel(c) > 1
                cc = [c(2:end), cc];
                c = c(1);
            end
            if any(strcmp(fieldnames(c.params), parm))
                % appending to an exissting parameter
                dim = find(strcmp(fieldnames(c.params), parm));
                sznew = size(c);
                sznew(dim) = 1;
                newdata = cellfun(@(x)reshape(x,sznew), {cc.data}, 'UniformOutput', false);
                c.data = cat(dim, c.data, newdata{:});
                c.params.(parm) = [c.params.(parm), val];
            else % create a new parameter
                % checking number of fields is better than nume(size) because of
                % the trailining dimensions of size 1
                c.data = cat(length(fieldnames(c.params))+1, c.data, cc.data);
                c.params.(parm) = val;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Arithmetic Operators
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function A = times(A, B)
            A.data = times(A.data, B);
        end
        function A = power(A,B)
            A.data = power(A.data, B);
        end
        function A = rdivide(A,B)
            A.data = rdivide(A.data, B);
        end
        function A = ldivide(A,B)
            A.data = ldivide(A.data, B);
        end
        function A = transpose(A)
            A.data = transpose(A.data);
        end
        function A = ctranspose(A)
            A.data = ctranspose(A.data);
        end
        function A = plus(A,B)
            A.data = plus(A.data, B);
        end
        function A = minus(A,B)
            A.data = minus(A.data, B);
        end
        function A = angle(A)
            A.data = angle(A.data);
        end
        function A = real(A)
            A.data = real(A.data);
        end
        function A = imag(A)
            A.data = imag(A.data);
        end
        function A = abs(A)
            A.data = abs(A.data);
        end
    end
end