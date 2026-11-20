# Uses the pinned SymbolicRegression 1.11.3 documented AbstractSRLogger extension.
# No Julia package or installed source is modified by this logger.
using Logging
mutable struct AcousticSnapshotLogger <: SymbolicRegression.AbstractSRLogger
    path::String
    t0::Float64
    last_time::Float64
    interval::Float64
    step::Int
end
SymbolicRegression.get_logger(::AcousticSnapshotLogger)=Logging.NullLogger()
function SymbolicRegression.SearchUtilsModule.logging_callback!(
    logger::AcousticSnapshotLogger; state,datasets,ropt,options)
    now=time();logger.step+=1
    if now-logger.last_time < logger.interval
        return nothing
    end
    logger.last_time=now
    members=SymbolicRegression.LoggingModule.calculate_pareto_frontier(state.halls_of_fame[1])
    open(logger.path,"a") do io
        for member in members
            equation=SymbolicRegression.LoggingModule.string_tree(member.tree,options;
                variable_names=datasets[1].variable_names,pretty=false,f_constant=string)
            equation=replace(equation,'\t'=>' ','\n'=>' ')
            println(io,join((now-logger.t0,now,logger.step,
                SymbolicRegression.compute_complexity(member,options),member.loss,
                state.cur_maxsizes[1],equation),'\t'))
        end
    end
    return nothing
end
function make_acoustic_logger(path,interval)
    open(path,"w") do io
        println(io,"elapsed_s\tunix_time\tstep\tcomplexity\ttraining_mse\teffective_maxsize\tequation")
    end
    now=time()
    AcousticSnapshotLogger(path,now,-Inf,interval,0)
end
