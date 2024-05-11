<?php

declare(strict_types=1);

namespace App\AI;

use App\Models\Agent;
use App\Models\NostrJob;
use App\Models\Thread;
use GuzzleHttp\Client;

class NostrInference
{
    public static function inference(string $model, NostrJob $job, callable $streamFunction, ?Client $httpClient = null): array
    {
        $thread = Thread::find($job->thread_id);
        $agent = Agent::find($job->agent_id);

        $prePrompt = 'You can use the following CONTEXT to help you answer the user\'s questions.';

        $prompt = $agent->prompt."\n".$prePrompt."\n".$job->content;

        return SimpleInferencer::inference($prompt, $model, $thread, $streamFunction, $httpClient);
    }
}
